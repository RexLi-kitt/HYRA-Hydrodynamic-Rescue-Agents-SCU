package reproduction;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.mojang.serialization.JsonOps;
import com.mojang.serialization.Lifecycle;
import etcodehome.freeterraforged.world.worldgen.FTFRandomState;
import net.minecraft.Util;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.core.MappedRegistry;
import net.minecraft.core.registries.Registries;
import net.minecraft.gametest.framework.GameTestServer;
import net.minecraft.gametest.framework.TestFunction;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.nbt.IntTag;
import net.minecraft.nbt.ListTag;
import net.minecraft.nbt.NbtIo;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.WorldLoader;
import net.minecraft.server.WorldStem;
import net.minecraft.server.packs.repository.PackRepository;
import net.minecraft.server.packs.repository.ServerPacksSource;
import net.minecraft.world.Difficulty;
import net.minecraft.world.flag.FeatureFlags;
import net.minecraft.world.level.DataPackConfig;
import net.minecraft.world.level.GameRules;
import net.minecraft.world.level.GameType;
import net.minecraft.world.level.LevelSettings;
import net.minecraft.world.level.WorldDataConfiguration;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.Rotation;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.dimension.LevelStem;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.WorldOptions;
import net.minecraft.world.level.levelgen.presets.WorldPresets;
import net.minecraft.world.level.storage.LevelResource;
import net.minecraft.world.level.storage.LevelStorageSource;
import net.minecraft.world.level.storage.PrimaryLevelData;
import net.minecraft.server.level.ServerLevel;

import java.lang.reflect.Constructor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/** Executes the entire real chunk pipeline, including surface, carvers and features. */
public final class FullWorldExporter {
    private static final Gson JSON = new GsonBuilder().setPrettyPrinting().create();

    public static void run(JsonObject request, Path output) throws Exception {
        if (!Files.isRegularFile(output.resolve("datapack/pack.mcmeta"))) {
            throw new IllegalArgumentException("Full world export requires export_datapack=true");
        }
        boolean surfaceOnly = request.has("surface");
        JsonObject region = request.getAsJsonObject(surfaceOnly ? "surface" : "voxels");
        boolean generateStructures = request.has("generate_structures")
                ? request.get("generate_structures").getAsBoolean()
                : !(request.has("natural_surface") && request.get("natural_surface").getAsBoolean());
        long seed = request.get("seed").getAsLong();
        LevelStorageSource storage = LevelStorageSource.createDefault(output.resolve("worlds"));
        var access = storage.createAccess("export-" + UUID.randomUUID());
        Path packs = access.getLevelPath(LevelResource.DATAPACK_DIR);
        Path pack = packs.resolve("original-terrain");
        copyTree(output.resolve("datapack"), pack);
        writeEmptyTemplate(pack);
        PackRepository repository = ServerPacksSource.createPackRepository(access);
        repository.reload();
        WorldDataConfiguration data = new WorldDataConfiguration(
                new DataPackConfig(enabledPacks(repository), List.of()),
                FeatureFlags.DEFAULT_FLAGS);
        GameRules rules = new GameRules();
        rules.getRule(GameRules.RULE_DOMOBSPAWNING).set(false, null);
        rules.getRule(GameRules.RULE_DAYLIGHT).set(false, null);
        rules.getRule(GameRules.RULE_RANDOMTICKING).set(0, null);
        LevelSettings settings = new LevelSettings("Original FreeTerraForged export", GameType.CREATIVE,
                false, Difficulty.PEACEFUL, true, rules, data);
        WorldLoader.InitConfig config = new WorldLoader.InitConfig(
                new WorldLoader.PackConfig(repository, data, false, false), Commands.CommandSelection.DEDICATED, 4);
        WorldStem stem = Util.blockUntilDone(executor -> WorldLoader.load(config, context -> {
            System.out.println("Loaded original noise settings: " + context.datapackWorldgen().registryOrThrow(Registries.NOISE_SETTINGS).getHolderOrThrow(net.minecraft.world.level.levelgen.NoiseGeneratorSettings.OVERWORLD).value().noiseSettings());
            System.out.println("Loaded dimension: " + context.datapackWorldgen().registryOrThrow(Registries.DIMENSION_TYPE).getHolderOrThrow(net.minecraft.world.level.dimension.BuiltinDimensionTypes.OVERWORLD).value().height());
            var emptyDimensions = new MappedRegistry<LevelStem>(Registries.LEVEL_STEM, Lifecycle.stable()).freeze();
            var dimensions = context.datapackWorldgen().registryOrThrow(Registries.WORLD_PRESET)
                    .getHolderOrThrow(WorldPresets.NORMAL).value().createWorldDimensions().bake(emptyDimensions);
            var worldData = new PrimaryLevelData(settings, new WorldOptions(seed, generateStructures, false),
                    dimensions.specialWorldProperty(), dimensions.lifecycle());
            return new WorldLoader.DataLoadOutput<>(worldData, dimensions.dimensionsRegistryAccess());
        }, WorldStem::new, Util.backgroundExecutor(), executor)).get();

        AtomicReference<Throwable> failure = new AtomicReference<>();
        var test = new TestFunction("original-terrain", "full-world-export", "reproduction:empty", Rotation.NONE,
                24000, 0, true, helper -> {
            try {
                if (surfaceOnly) SurfaceGridExporter.export(helper.getLevel(), region, output.resolve("surface"), seed, generateStructures);
                else exportBlocks(helper.getLevel(), region, output.resolve("voxels"), seed);
                helper.succeed();
            } catch (Throwable error) {
                failure.set(error);
                throw new RuntimeException("Full original chunk export failed", error);
            }
        });
        Constructor<GameTestServer> constructor = GameTestServer.class.getDeclaredConstructor(Thread.class,
                LevelStorageSource.LevelStorageAccess.class, PackRepository.class, WorldStem.class,
                Collection.class, BlockPos.class);
        constructor.setAccessible(true);
        // GameTestServer has no network listener or dedicated-server EULA acceptance step.
        // Use its unchanged implementation with a normal world instead of its default flat world.
        int step = region.has("step_blocks") ? region.get("step_blocks").getAsInt() : 1;
        int testX = Math.toIntExact((long) region.get("origin_x").getAsInt() + (long) region.get("nx").getAsInt() * step + 64);
        int testZ = Math.toIntExact((long) region.get("origin_z").getAsInt() + (long) region.get("nz").getAsInt() * step + 64);
        int worldHeight = etcodehome.freeterraforged.data.worldgen.preset.settings.Preset.DIRECT_CODEC
                .parse(JsonOps.INSTANCE, JSON.fromJson(Files.readString(output.resolve("preset.json")), JsonObject.class))
                .getOrThrow().world().properties.worldHeight;
        BlockPos testPosition = new BlockPos(testX, Math.min(300, worldHeight - 2), testZ);
        GameTestServer server = MinecraftServer.spin(thread -> {
            try {
                return constructor.newInstance(thread, access, repository, stem, List.of(test), testPosition);
            } catch (ReflectiveOperationException error) {
                throw new RuntimeException(error);
            }
        });
        server.getRunningThread().join();
        if (failure.get() != null) throw new RuntimeException(failure.get());
        if (!Files.isRegularFile(output.resolve(surfaceOnly ? "surface/surface.json" : "voxels/volume.json"))) throw new IllegalStateException("World export did not finish");
    }

    private static void exportBlocks(ServerLevel level, JsonObject region, Path output, long seed) throws Exception {
        var originalState = (FTFRandomState) (Object) level.getChunkSource().randomState();
        if (originalState.generatorContext() == null || originalState.preset() == null || originalState.seed() != seed) {
            throw new IllegalStateException("The original FreeTerraForged world generator is not active");
        }
        int nx = region.get("nx").getAsInt();
        int nz = region.get("nz").getAsInt();
        int originX = region.get("origin_x").getAsInt();
        int originZ = region.get("origin_z").getAsInt();
        int minY = region.has("min_y") ? region.get("min_y").getAsInt() : level.getMinBuildHeight();
        int maxY = region.has("max_y") ? region.get("max_y").getAsInt() : level.getMaxBuildHeight();
        if (nx < 1 || nz < 1 || minY < level.getMinBuildHeight() || maxY > level.getMaxBuildHeight() || minY >= maxY) {
            throw new IllegalArgumentException("Invalid voxel region");
        }
        int ny = maxY - minY;
        int count = Math.multiplyExact(Math.multiplyExact(nx, nz), ny);
        ByteBuffer blocks = ByteBuffer.allocate(Math.multiplyExact(count, 4)).order(ByteOrder.LITTLE_ENDIAN);
        ByteBuffer biomes = ByteBuffer.allocate(Math.multiplyExact(nx * nz, 4)).order(ByteOrder.LITTLE_ENDIAN);
        Map<Integer, JsonElement> palette = new TreeMap<>();
        Map<String, Integer> biomeIds = new LinkedHashMap<>();
        Map<Heightmap.Types, ByteBuffer> heights = new LinkedHashMap<>();
        for (Heightmap.Types type : List.of(Heightmap.Types.WORLD_SURFACE, Heightmap.Types.OCEAN_FLOOR,
                Heightmap.Types.MOTION_BLOCKING, Heightmap.Types.MOTION_BLOCKING_NO_LEAVES)) {
            heights.put(type, ByteBuffer.allocate(nx * nz * 4).order(ByteOrder.LITTLE_ENDIAN));
        }
        BlockPos.MutableBlockPos position = new BlockPos.MutableBlockPos();
        for (int z = 0; z < nz; z++) {
            for (int x = 0; x < nx; x++) {
                int worldX = Math.addExact(originX, x);
                int worldZ = Math.addExact(originZ, z);
                // This requests FULL status: all original noise, surface, carving and features run.
                var chunk = level.getChunk(Math.floorDiv(worldX, 16), Math.floorDiv(worldZ, 16));
                for (int y = minY; y < maxY; y++) {
                    position.set(worldX, y, worldZ);
                    BlockState state = chunk.getBlockState(position);
                    int id = Block.getId(state);
                    blocks.putInt(((z * ny + y - minY) * nx + x) * 4, id);
                    palette.computeIfAbsent(id, ignored -> BlockState.CODEC.encodeStart(JsonOps.INSTANCE, state).getOrThrow());
                }
                int index = z * nx + x;
                for (var height : heights.entrySet()) height.getValue().putInt(index * 4, chunk.getHeight(height.getKey(), worldX & 15, worldZ & 15));
                int surface = chunk.getHeight(Heightmap.Types.WORLD_SURFACE, worldX & 15, worldZ & 15);
                position.set(worldX, surface, worldZ);
                String biome = level.getBiome(position).unwrapKey().orElseThrow().location().toString();
                int biomeId = biomeIds.computeIfAbsent(biome, ignored -> biomeIds.size());
                biomes.putInt(index * 4, biomeId);
            }
            System.out.printf("Full original voxel rows: %d/%d%n", z + 1, nz);
        }
        Files.createDirectories(output);
        Files.write(output.resolve("blocks.i32"), blocks.array());
        Files.write(output.resolve("surface_biomes.i32"), biomes.array());
        for (var height : heights.entrySet()) Files.write(output.resolve(height.getKey().name().toLowerCase() + ".i32"), height.getValue().array());
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("upstream_revision", TerrainGridExporter.REVISION);
        metadata.put("pipeline", "Minecraft ServerLevel -> chunk status FULL with all original FreeTerraForged mixins");
        metadata.put("original_generator_context_initialized", true);
        metadata.put("original_preset_world_height", originalState.preset().world().properties.worldHeight);
        metadata.put("seed", Long.toString(seed));
        metadata.put("shape", new int[]{nz, ny, nx});
        metadata.put("axis_order", new String[]{"z", "y", "x"});
        metadata.put("origin_blocks", new int[]{originX, minY, originZ});
        metadata.put("step_blocks", 1);
        metadata.put("dtype", "<i4");
        metadata.put("block_state_palette", palette);
        metadata.put("surface_biome_ids", biomeIds);
        metadata.put("heightmaps", heights.keySet().stream().map(t -> t.name().toLowerCase() + ".i32").toList());
        Files.writeString(output.resolve("volume.json"), JSON.toJson(metadata) + "\n");
    }

    private static List<String> enabledPacks(PackRepository repository) {
        List<String> ids = new ArrayList<>();
        ids.add("vanilla");
        repository.getAvailableIds().stream()
                .filter(id -> !id.equals("vanilla") && !id.startsWith("file/") && !id.equals("bundle") && !id.equals("trade_rebalance"))
                .sorted().forEach(ids::add);
        ids.add("file/original-terrain");
        return ids;
    }

    private static void copyTree(Path source, Path target) throws Exception {
        try (var files = Files.walk(source)) {
            for (Path path : files.toList()) {
                Path destination = target.resolve(source.relativize(path));
                if (Files.isDirectory(path)) Files.createDirectories(destination);
                else Files.copy(path, destination, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    private static void writeEmptyTemplate(Path pack) throws Exception {
        CompoundTag tag = new CompoundTag();
        ListTag size = new ListTag();
        for (int n : new int[]{1, 1, 1}) size.add(IntTag.valueOf(n));
        tag.put("size", size);
        tag.put("palette", new ListTag());
        tag.put("blocks", new ListTag());
        tag.put("entities", new ListTag());
        tag.putInt("DataVersion", net.minecraft.SharedConstants.getCurrentVersion().getDataVersion().getVersion());
        Path file = pack.resolve("data/reproduction/structure/empty.nbt");
        Files.createDirectories(file.getParent());
        NbtIo.writeCompressed(tag, file);
    }
}
