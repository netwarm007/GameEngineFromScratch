#pragma once
#include <unordered_map>
#include "IShaderManager.hpp"

namespace std
{
	// Specialization for std::hash<Guid> -- this implementation
	// uses std::hash<std::string> on the stringification of the guid
	// to calculate the hash
	template <>
	struct hash<const My::DefaultShaderIndex>
	{
		using argument_type = My::DefaultShaderIndex;
		using result_type = std::size_t;

		result_type operator()(argument_type const &index) const
		{
			std::hash<My::IShaderManager::ShaderHandler> hasher;
			return static_cast<result_type>(hasher((My::IShaderManager::ShaderHandler)index));
		}
	};
}

namespace My {
    class ShaderManager : implements IShaderManager
    {
    public:
        ShaderManager() = default;
        ~ShaderManager() override = default;

        IShaderManager::ShaderHandler GetDefaultShaderProgram(DefaultShaderIndex index) final
        {
            return m_DefaultShaders[index];
        }

    protected:
        std::unordered_map<const DefaultShaderIndex, IShaderManager::ShaderHandler> m_DefaultShaders;
    };
}