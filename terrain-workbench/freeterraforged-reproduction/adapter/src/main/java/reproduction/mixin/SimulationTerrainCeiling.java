package reproduction.mixin;

import etcodehome.freeterraforged.world.worldgen.densityfunction.tile.filter.Filterable;
import etcodehome.freeterraforged.world.worldgen.densityfunction.tile.filter.TerrainCeiling;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import reproduction.SimulationHeightPolicy;

/** Simulation grids keep the generated elevations instead of compressing peaks to a game ceiling. */
@Mixin(value = TerrainCeiling.class, remap = false)
public abstract class SimulationTerrainCeiling {
    @Inject(method = "apply", at = @At("HEAD"), cancellable = true, remap = false)
    private void keepSimulationElevation(Filterable map, int seedX, int seedZ, int iterations, CallbackInfo callback) {
        if (SimulationHeightPolicy.unbounded()) callback.cancel();
    }
}
