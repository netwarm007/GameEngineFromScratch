#include "RhiStartup.hpp"

#include <array>
#include <stdexcept>
#include <utility>

namespace My {
namespace {
constexpr std::array backendNames = {
    std::pair{RhiBackend::D3d12, std::string_view{"d3d12"}},
    std::pair{RhiBackend::OpenGL, std::string_view{"opengl"}},
    std::pair{RhiBackend::Metal, std::string_view{"metal"}},
    std::pair{RhiBackend::Vulkan, std::string_view{"vulkan"}}};

}  // namespace

std::optional<RhiBackend> ParseRhiBackendName(std::string_view name) {
    for (const auto& [backend, canonicalName] : backendNames) {
        if (name == canonicalName) return backend;
    }
    return std::nullopt;
}

std::string_view RhiBackendName(RhiBackend backend) {
    for (const auto& [candidate, name] : backendNames) {
        if (candidate == backend) return name;
    }
    throw std::invalid_argument("Invalid RHI backend");
}

RhiStartupParseResult ParseRhiStartupArguments(
    std::span<char* const> arguments) {
    if (arguments.empty() || !arguments.front() || !*arguments.front()) {
        return {std::nullopt, "RHI argument parsing requires an executable name"};
    }
    for (const auto* argument : arguments) {
        if (!argument) {
            return {std::nullopt, "RHI argument parsing received a null argument"};
        }
    }

    RhiStartupOptions options;
    options.applicationArguments.reserve(arguments.size() + 1);
    options.applicationArguments.push_back(arguments.front());
    bool parseOptions = true;
    for (size_t i = 1; i < arguments.size(); ++i) {
        const std::string_view argument(arguments[i]);
        if (parseOptions && argument == "--") {
            parseOptions = false;
        } else if (parseOptions &&
                   (argument == "--rhi" || argument.starts_with("--rhi="))) {
            if (options.requestedBackend) {
                return {std::nullopt, "--rhi must be specified only once"};
            }
            std::string_view name;
            if (argument == "--rhi") {
                if (++i == arguments.size()) {
                    return {std::nullopt, "--rhi requires a backend name"};
                }
                name = arguments[i];
            } else {
                name = argument.substr(std::string_view("--rhi=").size());
            }
            if (name.empty()) {
                return {std::nullopt, "--rhi requires a backend name"};
            }
            options.requestedBackend = ParseRhiBackendName(name);
            if (!options.requestedBackend) {
                return {std::nullopt,
                        "Unknown RHI '" + std::string(name) +
                            "'; expected d3d12, opengl, metal, or vulkan"};
            }
            continue;
        } else if (parseOptions && argument == "--list-rhi") {
            options.listBackends = true;
            continue;
        } else if (parseOptions && argument.starts_with("--list-rhi=")) {
            return {std::nullopt, "--list-rhi does not take a value"};
        }
        options.applicationArguments.push_back(arguments[i]);
    }
    options.applicationArguments.push_back(nullptr);
    return {std::move(options), {}};
}

RhiStartupSelection ResolveRhiStartupBackend(
    std::optional<RhiBackend> requestedBackend, RhiBackend defaultBackend,
    std::span<const RhiBackendAvailability> availability) {
    const auto selected = requestedBackend.value_or(defaultBackend);
    const std::string prefix =
        std::string(requestedBackend ? "Requested RHI '" : "Default RHI '") +
        std::string(RhiBackendName(selected)) + "'";
    const RhiBackendAvailability* candidate = nullptr;
    for (const auto& entry : availability) {
        if (entry.backend != selected) continue;
        if (candidate) {
            return {std::nullopt, prefix + " has duplicate availability entries"};
        }
        candidate = &entry;
    }
    if (!candidate || candidate->state == RhiAvailability::NotCompiled) {
        std::string error = prefix + " is not compiled into this application";
        if (candidate && !candidate->reason.empty()) {
            error += ": " + candidate->reason;
        }
        return {std::nullopt, std::move(error)};
    }
    if (candidate->state == RhiAvailability::RuntimeUnavailable) {
        return {std::nullopt,
                prefix + " cannot initialize: " +
                    (candidate->reason.empty()
                         ? "runtime prerequisites unavailable"
                         : candidate->reason)};
    }
    if (candidate->state == RhiAvailability::ReadyForInitialization) {
        return {selected, {}};
    }
    throw std::invalid_argument("Invalid RHI availability state");
}

}  // namespace My
