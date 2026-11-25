#include "RhiStartup.hpp"

#include <array>
#include <initializer_list>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace My;

namespace {
int failures = 0;

void Expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

class Arguments {
   public:
    explicit Arguments(std::initializer_list<const char*> arguments) {
        for (const auto* argument : arguments) m_Strings.emplace_back(argument);
        for (auto& argument : m_Strings) values.push_back(argument.data());
    }

    std::vector<char*> values;

   private:
    std::vector<std::string> m_Strings;
};

void TestArgumentExtraction() {
    Arguments arguments{"Viewer", "--rhi", "vulkan", "-screenshot", "frame.png",
                        "--list-rhi", "Scene/billiard.ogex", "-sceneCycle", "2"};
    const auto parsed = ParseRhiStartupArguments(arguments.values);
    Expect(parsed.options.has_value() && parsed.error.empty(),
           "parse selection/list options mixed with application arguments");
    if (!parsed.options) return;
    Expect(parsed.options->requestedBackend == RhiBackend::Vulkan &&
               parsed.options->listBackends,
           "extract both RHI options");
    Expect(parsed.options->applicationArguments ==
               std::vector<char*>{arguments.values[0], arguments.values[3],
                                  arguments.values[4], arguments.values[6],
                                  arguments.values[7], arguments.values[8],
                                  nullptr},
           "retain executable, scene, screenshot and cycle arguments in order");

    const std::array names = {
        std::pair{"d3d12", RhiBackend::D3d12},
        std::pair{"opengl", RhiBackend::OpenGL},
        std::pair{"metal", RhiBackend::Metal},
        std::pair{"vulkan", RhiBackend::Vulkan}};
    for (const auto& [name, backend] : names) {
        const std::string option = std::string("--rhi=") + name;
        Arguments equalsArguments{"Viewer", option.c_str()};
        const auto equalsParsed =
            ParseRhiStartupArguments(equalsArguments.values);
        Expect(equalsParsed.options &&
                   equalsParsed.options->requestedBackend == backend,
               "accept each canonical name with equals syntax");
    }

    Arguments ordinary{"BilliardGame", "-physicsRegression", "captures"};
    const auto unchanged = ParseRhiStartupArguments(ordinary.values);
    Expect(unchanged.options && !unchanged.options->requestedBackend &&
               !unchanged.options->listBackends,
           "omitted options preserve application-defined default selection");
    if (unchanged.options) {
        auto expected = ordinary.values;
        expected.push_back(nullptr);
        Expect(unchanged.options->applicationArguments == expected,
               "ordinary arguments are unchanged");
    }

    Arguments terminated{"Viewer", "--rhi=opengl", "--", "--rhi=metal",
                         "--list-rhi", "Scene/billiard.ogex"};
    const auto afterTerminator = ParseRhiStartupArguments(terminated.values);
    Expect(afterTerminator.options &&
               afterTerminator.options->requestedBackend == RhiBackend::OpenGL &&
               !afterTerminator.options->listBackends,
           "do not consume RHI-looking application arguments after --");
    if (afterTerminator.options) {
        Expect(afterTerminator.options->applicationArguments ==
                   std::vector<char*>{terminated.values[0], terminated.values[2],
                                      terminated.values[3], terminated.values[4],
                                      terminated.values[5], nullptr},
               "forward the delimiter and its remainder unchanged");
    }
}

void ExpectParseError(std::initializer_list<const char*> values,
                      std::string_view diagnostic) {
    Arguments arguments(values);
    const auto parsed = ParseRhiStartupArguments(arguments.values);
    Expect(!parsed.options &&
               parsed.error.find(diagnostic) != std::string::npos,
           "reject invalid RHI arguments without publishing partial options");
}

void TestInvalidArguments() {
    ExpectParseError({"Viewer", "--rhi"}, "requires a backend name");
    ExpectParseError({"Viewer", "--rhi="}, "requires a backend name");
    ExpectParseError({"Viewer", "--rhi", ""}, "requires a backend name");
    ExpectParseError({"Viewer", "--rhi", "d2d"}, "Unknown RHI 'd2d'");
    ExpectParseError({"Viewer", "--rhi=Vulkan"}, "Unknown RHI 'Vulkan'");
    ExpectParseError({"Viewer", "--rhi=opengl", "--rhi", "metal"},
                     "only once");
    ExpectParseError({"Viewer", "--rhi", "opengl", "--rhi=opengl"},
                     "only once");
    ExpectParseError({"Viewer", "--list-rhi=yes"}, "does not take a value");
    ExpectParseError({}, "requires an executable name");
    ExpectParseError({""}, "requires an executable name");
    Arguments nullArgument{"Viewer", "Scene/billiard.ogex"};
    nullArgument.values[1] = nullptr;
    const auto parsed = ParseRhiStartupArguments(nullArgument.values);
    Expect(!parsed.options && !parsed.error.empty(),
           "reject a null argument rather than truncating the forwarded list");
}

void TestAvailabilityAndDefaults() {
    const std::array availability = {
        RhiBackendAvailability{RhiBackend::D3d12,
                               RhiAvailability::ReadyForInitialization, {}},
        RhiBackendAvailability{RhiBackend::OpenGL,
                               RhiAvailability::ReadyForInitialization, {}},
        RhiBackendAvailability{RhiBackend::Vulkan,
                               RhiAvailability::RuntimeUnavailable,
                               "Vulkan loader not found"},
        RhiBackendAvailability{RhiBackend::Metal,
                               RhiAvailability::NotCompiled,
                               "not a supported platform backend"}};
    const auto viewerDefault =
        ResolveRhiStartupBackend(std::nullopt, RhiBackend::D3d12, availability);
    const auto billiardDefault =
        ResolveRhiStartupBackend(std::nullopt, RhiBackend::OpenGL, availability);
    Expect(viewerDefault.backend == RhiBackend::D3d12 &&
               billiardDefault.backend == RhiBackend::OpenGL,
           "preserve distinct application defaults using the same contract");
    const auto explicitChoice =
        ResolveRhiStartupBackend(RhiBackend::OpenGL, RhiBackend::D3d12,
                                 availability);
    Expect(explicitChoice.backend == RhiBackend::OpenGL,
           "explicit selection overrides the default");

    const auto unavailable =
        ResolveRhiStartupBackend(RhiBackend::Vulkan, RhiBackend::OpenGL,
                                 availability);
    Expect(!unavailable.backend &&
               unavailable.error.find("Vulkan loader not found") !=
                   std::string::npos,
           "compiled backend with missing runtime fails without fallback");
    const auto unavailableDefault =
        ResolveRhiStartupBackend(std::nullopt, RhiBackend::Vulkan, availability);
    Expect(!unavailableDefault.backend &&
               unavailableDefault.error.find("Default RHI 'vulkan'") !=
                   std::string::npos,
           "unavailable default also fails despite ready alternatives");
    const auto notCompiled =
        ResolveRhiStartupBackend(RhiBackend::Metal, RhiBackend::OpenGL,
                                 availability);
    Expect(!notCompiled.backend &&
               notCompiled.error.find("not compiled") != std::string::npos &&
               notCompiled.error.find("not a supported platform backend") !=
                   std::string::npos,
           "report inclusion failure and retain its reason");
    const auto missing =
        ResolveRhiStartupBackend(RhiBackend::Vulkan, RhiBackend::OpenGL,
                                 std::span(availability).first(2));
    Expect(!missing.backend &&
               missing.error.find("not compiled") != std::string::npos,
           "omitted backend is not an implicit fallback");
    const auto empty =
        ResolveRhiStartupBackend(std::nullopt, RhiBackend::OpenGL, {});
    Expect(!empty.backend && !empty.error.empty(),
           "empty availability cannot publish a successful default");
    const std::array duplicates = {availability[0], availability[0]};
    const auto ambiguous =
        ResolveRhiStartupBackend(std::nullopt, RhiBackend::D3d12, duplicates);
    Expect(!ambiguous.backend &&
               ambiguous.error.find("duplicate") != std::string::npos,
           "reject conflicting registrations instead of using their order");
}
}  // namespace

int main() {
    TestArgumentExtraction();
    TestInvalidArguments();
    TestAvailabilityAndDefaults();
    if (failures) return 1;
    std::cout << "RHI startup contract tests passed\n";
    return 0;
}
