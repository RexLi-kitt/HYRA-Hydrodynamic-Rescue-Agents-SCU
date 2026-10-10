package reproduction;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.mojang.serialization.JsonOps;
import etcodehome.freeterraforged.config.PerformanceConfig;
import etcodehome.freeterraforged.data.worldgen.Datapacks;
import etcodehome.freeterraforged.data.worldgen.preset.settings.Preset;
import etcodehome.freeterraforged.data.worldgen.preset.settings.Presets;
import etcodehome.freeterraforged.registries.FTFRegistries;
import etcodehome.freeterraforged.world.worldgen.GeneratorContext;
import etcodehome.freeterraforged.world.worldgen.cell.Cell;
import etcodehome.freeterraforged.world.worldgen.cell.terrain.Terrain;
import etcodehome.freeterraforged.world.worldgen.densityfunction.tile.Tile;
import net.minecraft.core.HolderLookup;
import net.minecraft.data.CachedOutput;
import net.minecraft.data.DataProvider;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;

/** Sampling adapter only: all generation and filtering remain in upstream classes. */
public final class TerrainGridExporter implements DataProvider {
    private static final Gson JSON = new GsonBuilder().setPrettyPrinting().create();
    public static final String REVISION = "c5d322f5e44539ca6d3a6059c5de440af9abcfe6";
    private final CompletableFuture<HolderLookup.Provider> registries;

    public TerrainGridExporter(CompletableFuture<HolderLookup.Provider> registries) {
        this.registries = registries;
    }

    @Override
    public String getName() { return "FreeTerraForged original terrain grid export"; }

    @Override
    public CompletableFuture<?> run(CachedOutput ignored) {
        return registries.thenAccept(provider -> {
            try {
                Path request = Path.of(System.getProperty("reproduction.request"));
                JsonObject config = JSON.fromJson(Files.readString(request), JsonObject.class);
                if (config.has("validation") && config.get("validation").getAsBoolean()) {
                    ReproductionAcceptance.run(provider, request.toAbsolutePath().getParent());
                } else export(provider, config, request.toAbsolutePath().getParent());
            } catch (Exception error) {
                throw new RuntimeException("Original terrain export failed", error);
            } finally {
                // This standalone process has no game session to own the original worker pool.
                etcodehome.freeterraforged.concurrent.ThreadPools.WORLD_GEN.shutdown();
                etcodehome.freeterraforged.concurrent.cache.Cache.SCHEDULER.shutdown();
            }
        });
    }

    public static void export(HolderLookup.Provider vanilla, JsonObject request, Path directory) throws Exception {
        SimulationHeightPolicy.configure(request);
        long seed = request.get("seed").getAsLong();
        int nx = request.get("nx").getAsInt();
        int nz = request.get("nz").getAsInt();
        int step = request.get("step_blocks").getAsInt();
        int originX = request.get("origin_x").getAsInt();
        int originZ = request.get("origin_z").getAsInt();
        if (nx < 1 || nz < 1 || step < 1) throw new IllegalArgumentException("nx, nz, step_blocks must be positive");
        int count = Math.multiplyExact(nx, nz);
        Math.toIntExact((long) originX + (long) (nx - 1) * step);
        Math.toIntExact((long) originZ + (long) (nz - 1) * step);
        Path output = directory.resolve(request.get("output").getAsString()).normalize();
        Files.createDirectories(output);
        Preset preset = preset(request, directory);
        HolderLookup.Provider patched = preset.buildPatch(vanilla);
        PerformanceConfig performance = PerformanceConfig.makeDefault();
        int border = Math.min(2, Math.max(1, preset.filters().erosion.dropletLifetime / 16));
        GeneratorContext context = GeneratorContext.makeUncached(preset, patched.lookupOrThrow(FTFRegistries.NOISE),
                seed, performance.tileSize(), border, performance.batchCount());
        int tileWidth = context.generator.getTileBlockSize();

        List<Column> columns = new ArrayList<>();
        Field[] fields = Cell.class.getFields();
        Arrays.sort(fields, Comparator.comparing(Field::getName));
        for (Field field : fields) {
            if (!Modifier.isStatic(field.getModifiers())) columns.add(new Column(field, count));
        }
        record Key(int x, int z) implements Comparable<Key> {
            public int compareTo(Key other) { int c = Integer.compare(z, other.z); return c != 0 ? c : Integer.compare(x, other.x); }
        }
        Map<Key, List<Integer>> samples = new TreeMap<>();
        for (int row = 0; row < nz; row++) {
            int z = Math.toIntExact((long) originZ + (long) row * step);
            for (int col = 0; col < nx; col++) {
                int x = Math.toIntExact((long) originX + (long) col * step);
                samples.computeIfAbsent(new Key(Math.floorDiv(x, tileWidth), Math.floorDiv(z, tileWidth)), k -> new ArrayList<>())
                        .add(row * nx + col);
            }
        }
        double[] maximumHeight = {Double.NEGATIVE_INFINITY};
        int completed = 0;
        for (var group : samples.entrySet()) {
            Key key = group.getKey();
            try (Tile tile = context.generator.generate(key.x(), key.z()).join()) {
                for (Cell cell : tile.getBacking()) {
                    if (!Float.isFinite(cell.height)) throw new IllegalStateException("Non-finite terrain elevation");
                    maximumHeight[0] = Math.max(maximumHeight[0], (double) cell.height * context.levels.terrainScaleFactor);
                }
                for (int index : group.getValue()) {
                    int x = Math.toIntExact((long) originX + (long) (index % nx) * step);
                    int z = Math.toIntExact((long) originZ + (long) (index / nx) * step);
                    Cell cell = tile.lookup(x, z);
                    for (Column column : columns) column.store(index, cell);
                }
            }
            System.out.printf("Original terrain tiles: %d/%d%n", ++completed, samples.size());
        }

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("upstream_revision", REVISION);
        metadata.put("export_scope", "original Cell grid; request.voxels additionally runs full chunk pipeline");
        metadata.put("pipeline", "GeneratorContext -> TileGenerator.generate -> WorldFilters.apply(true)");
        metadata.put("seed", Long.toString(seed));
        metadata.put("shape", new int[]{nz, nx});
        metadata.put("axis_order", new String[]{"z", "x"});
        metadata.put("origin_blocks", new int[]{originX, originZ});
        metadata.put("step_blocks", step);
        metadata.put("tile_size_factor", performance.tileSize());
        metadata.put("tile_width_blocks", tileWidth);
        metadata.put("tile_border_chunks", border);
        metadata.put("terrain_scale_factor", context.levels.terrainScaleFactor);
        metadata.put("sea_level_blocks", context.levels.waterLevel);
        metadata.put("world_height_blocks", context.levels.worldHeight);
        metadata.put("world_depth_blocks", context.levels.worldDepth);
        metadata.put("height_ceiling_enabled", !SimulationHeightPolicy.unbounded() && preset.terrain().general.mountainVariety > 0.0F);
        metadata.put("height_policy", SimulationHeightPolicy.unbounded() ? "simulation_without_game_ceiling" : "original");
        Map<String, Object> schema = new LinkedHashMap<>();
        for (Column column : columns) {
            Files.write(output.resolve(column.filename()), column.buffer.array());
            Map<String, Object> descriptor = new LinkedHashMap<>();
            descriptor.put("file", column.filename());
            descriptor.put("dtype", column.dtype);
            if (!column.values.isEmpty()) descriptor.put("values", column.values);
            schema.put(column.field.getName(), descriptor);
        }
        metadata.put("fields", schema);
        Files.writeString(output.resolve("grid.json"), JSON.toJson(metadata) + "\n");
        Files.writeString(output.resolve("preset.json"), JSON.toJson(Preset.DIRECT_CODEC.encodeStart(JsonOps.INSTANCE, preset).getOrThrow()) + "\n");
        if (!request.has("export_datapack") || request.get("export_datapack").getAsBoolean()) {
            Preset surfacePreset = preset;
            if (SimulationHeightPolicy.unbounded() && (request.has("surface") || request.has("voxels"))) {
                // Reserve native block storage around the actual generated terrain; never clip the height grid.
                surfacePreset = preset.copy();
                int requiredTop = Math.toIntExact((long) Math.ceil((maximumHeight[0] + 64.0D) / 16.0D) * 16L);
                int top = Math.max(preset.world().properties.worldHeight, requiredTop);
                int depth = preset.world().properties.worldDepth;
                if (top - 1 > net.minecraft.world.level.dimension.DimensionType.MAX_Y
                        || (long) top + depth > net.minecraft.world.level.dimension.DimensionType.Y_SIZE) {
                    throw new IllegalArgumentException("Terrain exceeds native block engine capacity; the 2D height grid is preserved without clipping. Use quick terrain mode.");
                }
                surfacePreset.world().properties.worldHeight = top;
                Files.writeString(output.resolve("surface-preset.json"), JSON.toJson(Preset.DIRECT_CODEC.encodeStart(JsonOps.INSTANCE, surfacePreset).getOrThrow()) + "\n");
                metadata.put("surface_world_height_blocks", top);
                Files.writeString(output.resolve("grid.json"), JSON.toJson(metadata) + "\n");
            }
            Datapacks.makePreset(surfacePreset, vanilla, output.resolve("datapack-cache"), output.resolve("datapack"), "FreeTerraForged reproduction").run();
        }
        if (request.has("voxels") || request.has("surface")) FullWorldExporter.run(request, output);
    }

    private static Preset preset(JsonObject request, Path directory) throws IOException {
        if (request.has("preset_file")) {
            var tree = JSON.fromJson(Files.readString(directory.resolve(request.get("preset_file").getAsString())), com.google.gson.JsonElement.class);
            return configure(Preset.DIRECT_CODEC.parse(JsonOps.INSTANCE, tree).getOrThrow(), request);
        }
        Preset result = switch (request.has("preset") ? request.get("preset").getAsString() : "ftf-default") {
            case "ftf-default" -> Presets.makeFTFDefault();
            case "modern-default-with-rivers" -> Presets.modernDefaultWithRivers();
            case "legacy-default" -> Presets.makeLegacyDefault();
            case "legacy-vanillaish" -> Presets.makeLegacyVanillaish();
            case "legacy-beautiful" -> Presets.makeLegacyBeautiful();
            case "legacy-lite" -> Presets.makeLegacyLite();
            case "legacy-huge-biomes" -> Presets.makeLegacyHugeBiomes();
            case "community-1" -> Presets.makeCommunityPreset1();
            default -> throw new IllegalArgumentException("Unknown original preset");
        };
        if (request.has("continent_type")) result.world().continent.continentType = etcodehome.freeterraforged.data.worldgen.preset.settings.ContinentType.valueOf(request.get("continent_type").getAsString());
        return configure(result, request);
    }

    private static Preset configure(Preset preset, JsonObject request) {
        if (request.has("natural_surface") && request.get("natural_surface").getAsBoolean()) {
            // Configuration only: keep every original terrain algorithm unchanged.
            preset.terrain().volcano.weight = 0.0F;
            preset.terrain().general.fancyMountains = false;
            // The original archipelago climate branch can create mushroom islands.
            preset.island().enableArchipelago = false;
            preset.island().volcanoChance = 0.0F;
            preset.miscellaneous().volcanoBiomeUsage = 0.0F;
            preset.miscellaneous().vanillaLavaLakes = false;
            preset.miscellaneous().vanillaLavaSprings = false;
        }
        return preset;
    }

    private static final class Column {
        final Field field;
        final ByteBuffer buffer;
        final String dtype;
        final Map<Integer, String> values = new TreeMap<>();
        Column(Field field, int count) {
            this.field = field;
            Class<?> type = field.getType();
            dtype = type == float.class ? "<f4" : type == int.class || type == Terrain.class || type.isEnum() ? "<i4" : "|u1";
            if (!(type.isPrimitive() || type == Terrain.class || type.isEnum())) throw new IllegalArgumentException("Unhandled Cell field: " + field);
            buffer = ByteBuffer.allocate(Math.multiplyExact(count, dtype.endsWith("4") ? 4 : 1)).order(ByteOrder.LITTLE_ENDIAN);
        }
        String filename() { return field.getName() + (dtype.equals("<f4") ? ".f32" : dtype.equals("<i4") ? ".i32" : ".u8"); }
        void store(int index, Cell cell) throws IllegalAccessException {
            if (field.getType() == float.class) {
                float value = field.getFloat(cell);
                if (!Float.isFinite(value)) throw new IllegalStateException("Nonfinite " + field.getName() + " at " + index);
                buffer.putFloat(index * 4, value);
            } else if (field.getType() == int.class) buffer.putInt(index * 4, field.getInt(cell));
            else if (field.getType() == boolean.class) buffer.put(index, (byte) (field.getBoolean(cell) ? 1 : 0));
            else if (field.getType() == byte.class) buffer.put(index, field.getByte(cell));
            else if (field.get(cell) instanceof Terrain terrain) {
                buffer.putInt(index * 4, terrain.getId());
                values.put(terrain.getId(), terrain.getName());
            } else if (field.get(cell) instanceof Enum<?> value) {
                buffer.putInt(index * 4, value.ordinal());
                values.put(value.ordinal(), value.name());
            } else throw new IllegalStateException("Missing value: " + field.getName());
        }
    }
}
