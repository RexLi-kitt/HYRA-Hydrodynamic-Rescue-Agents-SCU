package reproduction;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import etcodehome.freeterraforged.config.PerformanceConfig;
import etcodehome.freeterraforged.data.worldgen.preset.settings.Presets;
import etcodehome.freeterraforged.registries.FTFRegistries;
import etcodehome.freeterraforged.world.worldgen.GeneratorContext;
import etcodehome.freeterraforged.world.worldgen.cell.Cell;
import etcodehome.freeterraforged.world.worldgen.cell.terrain.Terrain;
import net.minecraft.core.HolderLookup;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Integration checks against the original in-game cached Cell lookup. */
public final class ReproductionAcceptance {
    private static final Gson JSON = new GsonBuilder().setPrettyPrinting().create();

    public static void run(HolderLookup.Provider vanilla, Path directory) throws Exception {
        List<String> passed = new ArrayList<>();
        for (String name : List.of("ftf-default", "modern-default-with-rivers", "legacy-default", "legacy-vanillaish",
                "legacy-beautiful", "legacy-lite", "legacy-huge-biomes", "community-1")) {
            JsonObject request = request(name, "generated/acceptance/" + name);
            TerrainGridExporter.export(vanilla, request, directory);
            passed.add("preset:" + name);
        }
        JsonObject repeat = request("ftf-default", "generated/acceptance/repeat");
        TerrainGridExporter.export(vanilla, repeat, directory);
        Path first = directory.resolve("generated/acceptance/ftf-default");
        Path second = directory.resolve("generated/acceptance/repeat");
        JsonObject metadata = JSON.fromJson(Files.readString(first.resolve("grid.json")), JsonObject.class);
        for (var field : metadata.getAsJsonObject("fields").entrySet()) {
            String file = field.getValue().getAsJsonObject().get("file").getAsString();
            require(Arrays.equals(Files.readAllBytes(first.resolve(file)), Files.readAllBytes(second.resolve(file))), "Repeat differs: " + file);
        }
        passed.add("same-seed:all-34-fields-bit-identical");
        JsonObject different = request("ftf-default", "generated/acceptance/different-seed");
        different.addProperty("seed", "43");
        TerrainGridExporter.export(vanilla, different, directory);
        byte[] differentHeight = Files.readAllBytes(directory.resolve("generated/acceptance/different-seed/height.f32"));
        require(!Arrays.equals(Files.readAllBytes(first.resolve("height.f32")), differentHeight), "Seed has no effect");
        passed.add("different-seed:height-changes");
        for (String type : List.of("MULTI", "SINGLE", "MULTI_IMPROVED", "UPLIFT")) {
            JsonObject mode = request("ftf-default", "generated/acceptance/continent-" + type);
            mode.addProperty("continent_type", type);
            TerrainGridExporter.export(vanilla, mode, directory);
            passed.add("continent:" + type);
        }
        JsonObject roundTrip = request("ftf-default", "generated/acceptance/custom-json");
        roundTrip.addProperty("preset_file", "generated/acceptance/ftf-default/preset.json");
        TerrainGridExporter.export(vanilla, roundTrip, directory);
        byte[] roundTripHeight = Files.readAllBytes(directory.resolve("generated/acceptance/custom-json/height.f32"));
        require(Arrays.equals(Files.readAllBytes(first.resolve("height.f32")), roundTripHeight), "Preset JSON round trip differs");
        passed.add("custom-preset-json:height-identical");
        compareGameLookup(vanilla, first, metadata);
        passed.add("negative-coordinates-and-tile-boundaries:all-fields-match-original-game-lookup");
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("upstream_revision", TerrainGridExporter.REVISION);
        report.put("passed", passed);
        report.put("sample_coordinates", "5x5 grid from (-129,-129), step 65 blocks; crosses negative and positive tile boundaries");
        report.put("scope", "original Cell pipeline; full voxel pipeline validated separately");
        Files.createDirectories(directory.resolve("validation"));
        Files.writeString(directory.resolve("validation/integration.json"), JSON.toJson(report) + "\n");
    }

    private static JsonObject request(String preset, String output) {
        JsonObject request = new JsonObject();
        request.addProperty("seed", "42");
        request.addProperty("preset", preset);
        request.addProperty("origin_x", -129);
        request.addProperty("origin_z", -129);
        request.addProperty("nx", 5);
        request.addProperty("nz", 5);
        request.addProperty("step_blocks", 65);
        request.addProperty("output", output);
        request.addProperty("export_datapack", false);
        return request;
    }

    private static void compareGameLookup(HolderLookup.Provider vanilla, Path first, JsonObject metadata) throws Exception {
        var preset = Presets.makeFTFDefault();
        var patched = preset.buildPatch(vanilla);
        var performance = PerformanceConfig.makeDefault();
        GeneratorContext context = GeneratorContext.makeCached(preset, patched.lookupOrThrow(FTFRegistries.NOISE), 42,
                performance.tileSize(), performance.batchCount(), false);
        Map<String, ByteBuffer> data = new LinkedHashMap<>();
        for (var entry : metadata.getAsJsonObject("fields").entrySet()) {
            data.put(entry.getKey(), ByteBuffer.wrap(Files.readAllBytes(first.resolve(entry.getValue().getAsJsonObject().get("file").getAsString()))).order(ByteOrder.LITTLE_ENDIAN));
        }
        for (int row = 0; row < 5; row++) {
            for (int col = 0; col < 5; col++) {
                Cell cell = new Cell();
                require(context.lookup.applyCell(cell, -129 + col * 65, -129 + row * 65, true, true), "Original cache lookup failed");
                int index = row * 5 + col;
                for (var entry : data.entrySet()) {
                    var field = Cell.class.getField(entry.getKey());
                    ByteBuffer buffer = entry.getValue();
                    boolean equal;
                    if (field.getType() == float.class) equal = buffer.getInt(index * 4) == Float.floatToRawIntBits(field.getFloat(cell));
                    else if (field.getType() == int.class) equal = buffer.getInt(index * 4) == field.getInt(cell);
                    else if (field.getType() == byte.class) equal = buffer.get(index) == field.getByte(cell);
                    else if (field.getType() == boolean.class) equal = buffer.get(index) == (field.getBoolean(cell) ? 1 : 0);
                    else if (field.get(cell) instanceof Terrain terrain) equal = buffer.getInt(index * 4) == terrain.getId();
                    else equal = buffer.getInt(index * 4) == ((Enum<?>) field.get(cell)).ordinal();
                    require(equal, "Original lookup mismatch: " + entry.getKey() + " at " + index);
                }
            }
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}
