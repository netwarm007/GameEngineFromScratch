#include "GuiSubPass.hpp"
#include "imgui/imgui.h"
#include "flatbuffers/reflection.h"
#include "RenderDefinitions_generated.h"

#include <climits>
#include <cstdlib>
#include <cstdio>

using namespace My;

void GuiSubPass::Draw(Frame& frame) {
    if (ImGui::GetCurrentContext()) {
        ImGui::NewFrame();

        static Buffer bfbsfile;

        if (bfbsfile.GetDataSize() > 0) {
            static const reflection::Schema& schema = *reflection::GetSchema(bfbsfile.GetData());
            static auto enums = schema.enums();

            static std::vector<std::string> s_texture_formats_enum_data;
            static const char** s_texture_formats_enum_values;
            if (s_texture_formats_enum_data.size() == 0) {
                const reflection::Enum* enum_ = nullptr;

                enum_ = enums->LookupByKey("rendering.TextureFormat");

                if (enum_ != nullptr) {
                    auto values = enum_->values();

                    s_texture_formats_enum_data.reserve(values->size());
                    s_texture_formats_enum_values = new const char*[values->size()];

                    for (flatbuffers::uoffset_t i = 0; i < values->size(); i++) {
                        auto enum_value = values->Get(i);
                        s_texture_formats_enum_data.push_back(enum_value->name()->str());
                        s_texture_formats_enum_values[i] = s_texture_formats_enum_data.back().c_str();
                    }
                }
            }

            ImGui::Begin("Render Target", nullptr, ImGuiWindowFlags_AlwaysAutoResize
                                                            | ImGuiWindowFlags_NoFocusOnAppearing);

            auto types = schema.objects();
            auto render_target_type = types->LookupByKey("rendering.RenderTarget");
            auto fields = render_target_type->fields();
            static rendering::RenderTarget rt;
            flatbuffers::Struct* root = reinterpret_cast<flatbuffers::Struct*>(&rt);
            char svalue[128];
            if (fields) {
                for (flatbuffers::uoffset_t i = 0; i < fields->size(); i++) {
                    auto field = fields->Get(i);
                    auto field_base_type = field->type()->base_type();
                    auto field_attributes = field->attributes();
                    auto field_index = field->type()->index();
                    snprintf( svalue, 128, "%s", field->name()->c_str());

                    if (field_index >= 0) {
                        static int32_t s_data = 0;
                        int ivalue = static_cast<int>(flatbuffers::GetAnyFieldI(*root, *field));
                        ImGui::Combo(svalue, &ivalue, s_texture_formats_enum_values, s_texture_formats_enum_data.size());
                        flatbuffers::SetAnyFieldI(root, *field, ivalue);
                    } 
                    else {
                        switch (field_base_type) {
                            case reflection::BaseType::UShort:
                            {
                                int min_ = 0, max_ = UINT16_MAX;
                                if (field_attributes) {
                                    auto attr = field_attributes->LookupByKey("ui");
                                    if (attr) {
                                        std::sscanf(attr->value()->c_str(), "min:%d, max:%d", &min_, &max_);
                                    }
                                }
                                int ivalue = static_cast<int>(flatbuffers::GetAnyFieldI(*root, *field));
                                ImGui::SliderInt(svalue, &ivalue, min_, max_);
                                flatbuffers::SetAnyFieldI(root, *field, static_cast<int64_t>(ivalue));
                            }
                            break;
                            case reflection::BaseType::Float:
                            {
                                float min_ = 0.5f, max_ = 2.0f;
                                float fvalue = static_cast<float>(flatbuffers::GetAnyFieldF(*root, *field));
                                ImGui::SliderFloat(svalue, &fvalue, min_, max_);
                                flatbuffers::SetAnyFieldF(root, *field, static_cast<double>(fvalue));
                            }
                            break;
                            default:
                                ;
                        }
                    }
                }
            }

            fprintf(stderr, "RenderTarget: width = %d, height = %d\n, scale_x = %f, scale_y = %f, format = %s\n", 
                rt.width(), rt.height(), rt.scale_x(), rt.scale_y(), 
                rendering::EnumNamesTextureFormat()[static_cast<int>(rt.format())]);

            ImGui::End();
        }
        else {
            bfbsfile = g_pAssetLoader->SyncOpenAndReadBinary("Data/RenderDefinitions.bfbs");
        }

        ImGui::Render();

        ImGui::EndFrame();

    }
}