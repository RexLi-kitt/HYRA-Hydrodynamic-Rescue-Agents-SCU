package reproduction.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** CLI lifecycle only; waits until every datagen provider and cache write has finished. */
@Mixin(value = net.fabricmc.fabric.impl.datagen.FabricDataGenHelper.class, remap = false)
public abstract class DataGenerationExit {
    @Inject(method = "run", at = @At("RETURN"), remap = false)
    private static void exitCompletedExport(CallbackInfo callback) {
        if (System.getProperty("reproduction.request") != null) System.exit(0);
    }
}
