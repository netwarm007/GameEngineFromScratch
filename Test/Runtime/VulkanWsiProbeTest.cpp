#include "RhiStartup.hpp"
#include "VulkanLoader.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

void SetFixtureMode(const char* mode) {
#ifdef _WIN32
    _putenv_s("GEFS_FIXTURE_WSI_MODE", mode);
#else
    setenv("GEFS_FIXTURE_WSI_MODE", mode, 1);
#endif
}

RhiBackendAvailability ProbeMode(const char* fixture, const char* mode) {
    SetFixtureMode(mode);
    VulkanLoader loader(fixture);
    auto availability = ProbeVulkanWsiAvailability(loader);
    const auto balance = reinterpret_cast<int (*)()>(
        loader.GetExportedFunction("GEFS_FixtureInstanceBalance"));
    Expect(balance && balance() == 0,
           "probe destroys every instance it creates");
    std::cout << "mode " << mode << ": state="
              << static_cast<int>(availability.state) << " reason=["
              << availability.reason << "]\n";
    return availability;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "VulkanWsiProbeTest requires the WSI probe fixture library\n";
        return 2;
    }
    const char* fixture = argv[1];

    // The fixture offers only extension strings whose guards exist in the
    // compiled headers (KHR maintenance1 guards arrived after EXT; e.g.
    // 1.4.315 headers expose only EXT). Expectations follow the same gates.
    int macroStatus = 0;
    {
        VulkanLoader loader(fixture);
        const auto macros = reinterpret_cast<int (*)()>(
            loader.GetExportedFunction("GEFS_FixtureWsiMacroStatus"));
        Expect(macros != nullptr, "fixture reports its WSI macro status");
        if (macros) macroStatus = macros();
    }
    const bool anyMaintenance = (macroStatus & 3) != 0;
    std::cout << "fixture macros: KHR=" << ((macroStatus & 1) ? "defined" : "undefined")
              << " EXT=" << ((macroStatus & 2) ? "defined" : "undefined") << '\n';

    const auto expectReadyOrHeaderGate =
        [&](const RhiBackendAvailability& availability, const char* message) {
            if (anyMaintenance) {
                Expect(availability.state == RhiAvailability::ReadyForInitialization &&
                           availability.reason.empty(),
                       message);
            } else {
                Expect(availability.state == RhiAvailability::RuntimeUnavailable &&
                           availability.reason.find("SDK headers") != std::string::npos,
                       message);
            }
        };
    const auto expectReject = [&](const RhiBackendAvailability& availability,
                                  const char* needle, const char* message) {
        Expect(availability.state == RhiAvailability::RuntimeUnavailable &&
                   availability.reason.find(anyMaintenance ? needle : "SDK headers") !=
                       std::string::npos,
               message);
    };

    const auto full = ProbeMode(fixture, "khr");
    expectReadyOrHeaderGate(full,
                            "full maintenance1 support is ready to attempt initialization");

    const auto extPair = ProbeMode(fixture, "ext");
    expectReadyOrHeaderGate(extPair,
                            "ext-mode maintenance1 pair is accepted like the native gate");

    const auto noSurface = ProbeMode(fixture, "no-surface-maintenance");
    expectReject(noSurface, "surface_maintenance1",
                 "missing surface-maintenance1 rejects with a named reason");

    const auto noCapabilities2 = ProbeMode(fixture, "no-cap2");
    expectReject(noCapabilities2, "get_surface_capabilities2",
                 "missing get_surface_capabilities2 rejects with a named reason");

    const auto noSwapchain = ProbeMode(fixture, "no-swapchain-maintenance");
    expectReject(noSwapchain, "swapchain-maintenance1",
                 "missing device swapchain-maintenance1 rejects with a named reason");

    const auto featureOff = ProbeMode(fixture, "feature-off");
    expectReject(featureOff, "swapchain-maintenance1",
                 "extensions without the swapchainMaintenance1 feature reject");

#ifdef __APPLE__
    const auto portability = ProbeMode(fixture, "portability-required");
    expectReadyOrHeaderGate(portability,
                            "portability ICD is enumerated like the native Apple gate");
#endif

    if (failures) return 1;
    std::cout << "Vulkan WSI probe tests passed without a Vulkan driver\n";
    return 0;
}
