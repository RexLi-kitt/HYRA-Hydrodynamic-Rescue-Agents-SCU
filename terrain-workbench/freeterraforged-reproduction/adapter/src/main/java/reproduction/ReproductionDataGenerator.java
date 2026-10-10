package reproduction;

import net.fabricmc.fabric.api.datagen.v1.DataGeneratorEntrypoint;
import net.fabricmc.fabric.api.datagen.v1.FabricDataGenerator;

/** Runs inside Fabric's real, mixin-enabled Minecraft data-generation runtime. */
public final class ReproductionDataGenerator implements DataGeneratorEntrypoint {
    @Override
    public void onInitializeDataGenerator(FabricDataGenerator generator) {
        generator.createPack().addProvider((output, registries) -> new TerrainGridExporter(registries));
    }
}
