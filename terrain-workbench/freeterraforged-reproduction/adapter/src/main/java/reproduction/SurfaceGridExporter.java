package reproduction;

import com.google.gson.GsonBuilder;
import etcodehome.freeterraforged.world.worldgen.FTFRandomState;
import com.google.gson.JsonObject;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.tags.BlockTags;
import net.minecraft.tags.FluidTags;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.levelgen.Heightmap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/** Actual surface sampling after the unchanged FULL chunk pipeline. */
public final class SurfaceGridExporter {
    public static void export(ServerLevel level, JsonObject region, Path output, long seed, boolean structures) throws Exception {
        var original = (FTFRandomState) (Object) level.getChunkSource().randomState();
        if (original.generatorContext() == null || original.preset() == null || original.seed() != seed) {
            throw new IllegalStateException("Original FreeTerraForged generator is not active");
        }
        int nx = region.get("nx").getAsInt();
        int nz = region.get("nz").getAsInt();
        int step = region.has("step_blocks") ? region.get("step_blocks").getAsInt() : 1;
        int ox = region.get("origin_x").getAsInt(), oz = region.get("origin_z").getAsInt();
        if (nx < 1 || nz < 1 || step < 1) throw new IllegalArgumentException("Invalid surface grid");
        int count = Math.multiplyExact(nx, nz);
        Map<String, ByteBuffer> fields = new LinkedHashMap<>();
        for (String field : new String[]{"ground_height", "ground_block_id", "biome_id", "water_surface", "water_depth", "cover_flags", "visible_height"}) {
            fields.put(field, ByteBuffer.allocate(Math.multiplyExact(count, 4)).order(ByteOrder.LITTLE_ENDIAN));
        }
        Map<String, Integer> blocks = new LinkedHashMap<>(), biomes = new LinkedHashMap<>();
        BlockPos.MutableBlockPos pos = new BlockPos.MutableBlockPos();
        for (int z = 0; z < nz; z++) {
            for (int x = 0; x < nx; x++) {
                int wx = Math.toIntExact((long) ox + (long) x * step), wz = Math.toIntExact((long) oz + (long) z * step);
                var chunk = level.getChunk(Math.floorDiv(wx, 16), Math.floorDiv(wz, 16));
                int top = chunk.getHeight(Heightmap.Types.WORLD_SURFACE, wx & 15, wz & 15);
                int flags = 0, ground = level.getMinBuildHeight();
                float water = Float.NaN;
                BlockState groundState = null;
                for (int y = top - 1; y >= level.getMinBuildHeight(); y--) {
                    pos.set(wx, y, wz);
                    BlockState state = chunk.getBlockState(pos);
                    if (state.getFluidState().is(FluidTags.WATER)) {
                        water = Float.isNaN(water) ? y + state.getFluidState().getHeight(level, pos) : water;
                    }
                    if (state.is(BlockTags.LEAVES) || state.is(BlockTags.LOGS)) { flags |= 1; continue; }
                    if (state.is(Blocks.MANGROVE_ROOTS) || state.is(Blocks.MUDDY_MANGROVE_ROOTS)) { flags |= 1; continue; }
                    if (state.is(Blocks.CACTUS) || state.is(Blocks.BAMBOO) || state.is(Blocks.BAMBOO_SAPLING)) { flags |= 2; continue; }
                    if (state.is(Blocks.SNOW) || state.is(Blocks.POWDER_SNOW)) { flags |= 4; continue; }
                    if (state.is(Blocks.ICE) || state.is(Blocks.PACKED_ICE) || state.is(Blocks.BLUE_ICE) || state.is(Blocks.FROSTED_ICE)) { flags |= 8; continue; }
                    if (state.is(Blocks.MUSHROOM_STEM) || state.is(Blocks.RED_MUSHROOM_BLOCK) || state.is(Blocks.BROWN_MUSHROOM_BLOCK)) { flags |= 2; continue; }
                    if (!state.isAir() && state.getFluidState().isEmpty() && !state.blocksMotion()) flags |= 2;
                    if (state.blocksMotion() && !state.liquid()) {
                        ground = y + 1;
                        groundState = state;
                        break;
                    }
                }
                if (groundState == null) throw new IllegalStateException("No ground at " + wx + "," + wz);
                String block = BuiltInRegistries.BLOCK.getKey(groundState.getBlock()).toString();
                int blockId = blocks.computeIfAbsent(block, ignored -> blocks.size());
                pos.set(wx, Float.isNaN(water) ? ground : Math.max(ground, (int) Math.ceil(water)), wz);
                String biome = level.getBiome(pos).unwrapKey().orElseThrow().location().toString();
                int biomeId = biomes.computeIfAbsent(biome, ignored -> biomes.size());
                int offset = (z * nx + x) * 4;
                fields.get("ground_height").putInt(offset, ground);
                fields.get("ground_block_id").putInt(offset, blockId);
                fields.get("biome_id").putInt(offset, biomeId);
                fields.get("water_surface").putFloat(offset, water);
                fields.get("water_depth").putFloat(offset, Float.isNaN(water) ? 0.0F : Math.max(0.0F, water - ground));
                fields.get("cover_flags").putInt(offset, flags);
                fields.get("visible_height").putInt(offset, top);
            }
            if (z % 16 == 15 || z == nz - 1) System.out.printf("Original surface rows: %d/%d%n", z + 1, nz);
        }
        Files.createDirectories(output);
        Map<String, Object> schema = new LinkedHashMap<>();
        for (var field : fields.entrySet()) {
            boolean isFloat = field.getKey().equals("water_surface") || field.getKey().equals("water_depth");
            String filename = field.getKey() + (isFloat ? ".f32" : ".i32");
            Files.write(output.resolve(filename), field.getValue().array());
            schema.put(field.getKey(), Map.of("file", filename, "dtype", isFloat ? "<f4" : "<i4"));
        }
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("upstream_revision", TerrainGridExporter.REVISION);
        metadata.put("pipeline", "Actual ServerLevel FULL chunks, original FreeTerraForged generation");
        metadata.put("original_generator_context_initialized", true);
        metadata.put("structures_enabled", structures);
        metadata.put("height_policy", SimulationHeightPolicy.unbounded() ? "simulation_without_game_ceiling" : "original");
        metadata.put("world_min_y", level.getMinBuildHeight());
        metadata.put("world_max_y_exclusive", level.getMaxBuildHeight());
        metadata.put("seed", Long.toString(seed));
        metadata.put("shape", new int[]{nz, nx});
        metadata.put("axis_order", new String[]{"z", "x"});
        metadata.put("origin_blocks", new int[]{ox, oz});
        metadata.put("step_blocks", step);
        metadata.put("ground_height_definition", "Top face y+1 of first motion-blocking terrain block below vegetation, fluid, snow layers and ice; includes ground snow blocks. Not canopy height.");
        metadata.put("water_surface_definition", "Highest actual water fluid top above this ground; NaN on dry cells. Generated initial water, not a flood simulation.");
        metadata.put("cover_flags", Map.of("tree", 1, "other_vegetation", 2, "snow_layer_or_powder", 4, "ice", 8));
        metadata.put("block_ids", blocks);
        metadata.put("biome_ids", biomes);
        metadata.put("fields", schema);
        Files.writeString(output.resolve("surface.json"), new GsonBuilder().setPrettyPrinting().create().toJson(metadata) + "\n");
    }
}
