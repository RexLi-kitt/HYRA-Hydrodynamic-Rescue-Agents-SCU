package reproduction;

import com.google.gson.JsonObject;

/** Per-export policy; native reproduction remains the default outside the simulation UI. */
public final class SimulationHeightPolicy {
    private static volatile boolean unbounded;

    private SimulationHeightPolicy() {}

    public static void configure(JsonObject request) {
        unbounded = request.has("unbounded_height") && request.get("unbounded_height").getAsBoolean();
    }

    public static boolean unbounded() { return unbounded; }
}
