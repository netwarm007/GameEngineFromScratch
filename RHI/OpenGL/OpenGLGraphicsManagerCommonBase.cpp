// This is a common code snippet
// should be included in other source
// other than compile it independently

void OpenGLGraphicsManagerCommonBase::Clear()
{
    GraphicsManager::Clear();

    // Set the color to clear the screen to.
    glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
    // Clear the screen and depth buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLGraphicsManagerCommonBase::Finalize()
{
    GraphicsManager::Finalize();
}

void OpenGLGraphicsManagerCommonBase::Draw()
{
    GraphicsManager::Draw();

    glFlush();
}

bool OpenGLGraphicsManagerCommonBase::SetPerFrameShaderParameters(const DrawFrameContext& context)
{
    bool result;
    unsigned int location;

    // Set the world matrix in the vertex shader.
    result = SetShaderParameter("worldMatrix", context.m_worldMatrix);
    if (!result) return result;

    // Set the view matrix in the vertex shader.
    result = SetShaderParameter("viewMatrix", context.m_viewMatrix);
    if (!result) return result;

    // Set the projection matrix in the vertex shader.
    result = SetShaderParameter("projectionMatrix", context.m_projectionMatrix);
    if (!result) return result;

    // Set the ambient color
    result = SetShaderParameter("ambientColor", context.m_ambientColor);
    if (!result) return result;

    // Set number of lights
    result = SetShaderParameter("numLights", (int32_t)context.m_lights.size());
    if (!result) return result;

    // Set lighting parameters for PS shader
    for (size_t i = 0; i < context.m_lights.size(); i++)
    {
        ostringstream ss;
        string uniformName;

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightPosition" << ends;
        uniformName = ss.str();
        result = SetShaderParameter(uniformName.c_str(), context.m_lights[i].m_lightPosition);
        if (!result) return result;

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightColor" << ends;
        uniformName = ss.str();
        result = SetShaderParameter(uniformName.c_str(), context.m_lights[i].m_lightColor);
        if (!result) return result;

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightIntensity" << ends;
        uniformName = ss.str();
        result = SetShaderParameter(uniformName.c_str(), context.m_lights[i].m_lightIntensity);
        if (!result) return result;

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightDirection" << ends;
        uniformName = ss.str();
        result = SetShaderParameter(uniformName.c_str(), context.m_lights[i].m_lightDirection);
        if (!result) return result;

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightSize" << ends;
        uniformName = ss.str();
        result = SetShaderParameter(uniformName.c_str(), context.m_lights[i].m_lightSize);
        if (!result) return result;

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightDistAttenCurveType" << ends;
        uniformName = ss.str();
        result = SetShaderParameter(uniformName.c_str(), (int32_t)context.m_lights[i].m_lightDistAttenCurveType);
        if (!result) return result;

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightDistAttenCurveParams" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(m_CurrentShader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform1fv(location, 5, context.m_lights[i].m_lightDistAttenCurveParams);

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightAngleAttenCurveType" << ends;
        uniformName = ss.str();
        result = SetShaderParameter(uniformName.c_str(), (int32_t)context.m_lights[i].m_lightAngleAttenCurveType);
        if (!result) return result;

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightAngleAttenCurveParams" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(m_CurrentShader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform1fv(location, 5, context.m_lights[i].m_lightAngleAttenCurveParams);
    }

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetShaderParameter(const char* paramName, const Matrix4X4f& param)
{
    unsigned int location;

    location = glGetUniformLocation(m_CurrentShader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniformMatrix4fv(location, 1, false, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetShaderParameter(const char* paramName, const Vector2f& param)
{
    unsigned int location;

    location = glGetUniformLocation(m_CurrentShader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform2fv(location, 1, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetShaderParameter(const char* paramName, const Vector3f& param)
{
    unsigned int location;

    location = glGetUniformLocation(m_CurrentShader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform3fv(location, 1, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetShaderParameter(const char* paramName, const Vector4f& param)
{
    unsigned int location;

    location = glGetUniformLocation(m_CurrentShader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform4fv(location, 1, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetShaderParameter(const char* paramName, const float param)
{
    unsigned int location;

    location = glGetUniformLocation(m_CurrentShader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform1f(location, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetShaderParameter(const char* paramName, const int param)
{
    unsigned int location;

    location = glGetUniformLocation(m_CurrentShader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform1i(location, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetShaderParameter(const char* paramName, const bool param)
{
    unsigned int location;

    location = glGetUniformLocation(m_CurrentShader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform1f(location, param);

    return true;
}

void OpenGLGraphicsManagerCommonBase::InitializeBuffers(const Scene& scene)
{
    // Geometries
    for (auto _it : scene.GeometryNodes)
    {
        auto pGeometryNode = _it.second.lock();
        if (pGeometryNode && pGeometryNode->Visible()) 
        {
            auto pGeometry = scene.GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);
            auto pMesh = pGeometry->GetMesh().lock();
            if (!pMesh) continue;

            // Set the number of vertex properties.
            auto vertexPropertiesCount = pMesh->GetVertexPropertiesCount();

            // Allocate an OpenGL vertex array object.
            GLuint vao;
            glGenVertexArrays(1, &vao);

            // Bind the vertex array object to store all the buffers and vertex attributes we create here.
            glBindVertexArray(vao);

            GLuint buffer_id;

            for (uint32_t i = 0; i < vertexPropertiesCount; i++)
            {
                const SceneObjectVertexArray& v_property_array = pMesh->GetVertexPropertyArray(i);
                auto v_property_array_data_size = v_property_array.GetDataSize();
                auto v_property_array_data = v_property_array.GetData();

                // Generate an ID for the vertex buffer.
                glGenBuffers(1, &buffer_id);

                // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
                glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
                glBufferData(GL_ARRAY_BUFFER, v_property_array_data_size, v_property_array_data, GL_STATIC_DRAW);

                glEnableVertexAttribArray(i);

                switch (v_property_array.GetDataType()) {
                    case VertexDataType::kVertexDataTypeFloat1:
                        glVertexAttribPointer(i, 1, GL_FLOAT, false, 0, 0);
                        break;
                    case VertexDataType::kVertexDataTypeFloat2:
                        glVertexAttribPointer(i, 2, GL_FLOAT, false, 0, 0);
                        break;
                    case VertexDataType::kVertexDataTypeFloat3:
                        glVertexAttribPointer(i, 3, GL_FLOAT, false, 0, 0);
                        break;
                    case VertexDataType::kVertexDataTypeFloat4:
                        glVertexAttribPointer(i, 4, GL_FLOAT, false, 0, 0);
                        break;
#ifndef OPENGL_ES
                    case VertexDataType::kVertexDataTypeDouble1:
                        glVertexAttribPointer(i, 1, GL_DOUBLE, false, 0, 0);
                        break;
                    case VertexDataType::kVertexDataTypeDouble2:
                        glVertexAttribPointer(i, 2, GL_DOUBLE, false, 0, 0);
                        break;
                    case VertexDataType::kVertexDataTypeDouble3:
                        glVertexAttribPointer(i, 3, GL_DOUBLE, false, 0, 0);
                        break;
                    case VertexDataType::kVertexDataTypeDouble4:
                        glVertexAttribPointer(i, 4, GL_DOUBLE, false, 0, 0);
                        break;
#endif
                    default:
                        assert(0);
                }

                m_Buffers.push_back(buffer_id);
            }

            auto indexGroupCount = pMesh->GetIndexGroupCount();

            GLenum  mode;
            switch(pMesh->GetPrimitiveType())
            {
                case PrimitiveType::kPrimitiveTypePointList:
                    mode = GL_POINTS;
                    break;
                case PrimitiveType::kPrimitiveTypeLineList:
                    mode = GL_LINES;
                    break;
                case PrimitiveType::kPrimitiveTypeLineStrip:
                    mode = GL_LINE_STRIP;
                    break;
                case PrimitiveType::kPrimitiveTypeTriList:
                    mode = GL_TRIANGLES;
                    break;
                case PrimitiveType::kPrimitiveTypeTriStrip:
                    mode = GL_TRIANGLE_STRIP;
                    break;
                case PrimitiveType::kPrimitiveTypeTriFan:
                    mode = GL_TRIANGLE_FAN;
                    break;
                default:
                    // ignore
                    continue;
            }

            for (decltype(indexGroupCount) i = 0; i < indexGroupCount; i++)
            {
                // Generate an ID for the index buffer.
                glGenBuffers(1, &buffer_id);

                const SceneObjectIndexArray& index_array      = pMesh->GetIndexArray(i);
                auto index_array_size = index_array.GetDataSize();
                auto index_array_data = index_array.GetData();

                // Bind the index buffer and load the index data into it.
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer_id);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_array_size, index_array_data, GL_STATIC_DRAW);

                // Set the number of indices in the index array.
                GLsizei indexCount = static_cast<GLsizei>(index_array.GetIndexCount());
                GLenum type;
                switch(index_array.GetIndexType())
                {
                    case IndexDataType::kIndexDataTypeInt8:
                        type = GL_UNSIGNED_BYTE;
                        break;
                    case IndexDataType::kIndexDataTypeInt16:
                        type = GL_UNSIGNED_SHORT;
                        break;
                    case IndexDataType::kIndexDataTypeInt32:
                        type = GL_UNSIGNED_INT;
                        break;
                    default:
                        // not supported by OpenGL
                        cerr << "Error: Unsupported Index Type " << index_array << endl;
                        cerr << "Mesh: " << *pMesh << endl;
                        cerr << "Geometry: " << *pGeometry << endl;
                        continue;
                }

                m_Buffers.push_back(buffer_id);

                size_t material_index = index_array.GetMaterialIndex();
                std::string material_key = pGeometryNode->GetMaterialRef(material_index);
                auto material = scene.GetMaterial(material_key);
                if (material) {
                    auto color = material->GetBaseColor();
                    if (color.ValueMap) {
                        Image texture = color.ValueMap->GetTextureImage();
                        auto it = m_TextureIndex.find(material_key);
                        if (it == m_TextureIndex.end()) {
                            GLuint texture_id;
                            glGenTextures(1, &texture_id);
                            glActiveTexture(GL_TEXTURE0 + texture_id);
                            glBindTexture(GL_TEXTURE_2D, texture_id);
                            if(texture.bitcount == 24)
                            {
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture.Width, texture.Height, 
                                    0, GL_RGB, GL_UNSIGNED_BYTE, texture.data);
                            }
                            else
                            {
                                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.Width, texture.Height, 
                                    0, GL_RGBA, GL_UNSIGNED_BYTE, texture.data);
                            }
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

                            m_TextureIndex[color.ValueMap->GetName()] = texture_id;
                            m_Textures.push_back(texture_id);
                        }
                    }
                }

                auto dbc = make_shared<OpenGLDrawBatchContext>();
                dbc->vao     = vao;
                dbc->mode    = mode;
                dbc->type    = type;
                dbc->count   = indexCount;
                dbc->node    = pGeometryNode;
                dbc->material = material;
                m_Frames[m_nFrameIndex].batchContexts.push_back(dbc);
            }
        }
    }

    return;
}

void OpenGLGraphicsManagerCommonBase::ClearBuffers()
{
    for (int i = 0; i < kFrameCount; i++)
    {
        auto& batchContexts = m_Frames[i].batchContexts;

        for (auto dbc : batchContexts) {
            glDeleteVertexArrays(1, &dynamic_pointer_cast<OpenGLDrawBatchContext>(dbc)->vao);
        }

        batchContexts.clear();
    }

    for (auto buf : m_Buffers) {
        glDeleteBuffers(1, &buf);
    }

    for (auto texture : m_Textures) {
        glDeleteTextures(1, &texture);
    }

    m_Buffers.clear();
    m_Textures.clear();

}

void OpenGLGraphicsManagerCommonBase::UseShaderProgram(const intptr_t shaderProgram)
{
    m_CurrentShader = static_cast<GLuint>(shaderProgram);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    glUseProgram(m_CurrentShader);
}

void OpenGLGraphicsManagerCommonBase::SetPerFrameConstants(const DrawFrameContext& context)
{
    bool result = SetPerFrameShaderParameters(context);
    assert(result);
}

void OpenGLGraphicsManagerCommonBase::DrawBatch(const DrawBatchContext& context)
{
    const OpenGLDrawBatchContext& dbc = dynamic_cast<const OpenGLDrawBatchContext&>(context);

    bool result = SetShaderParameter("modelMatrix", dbc.trans);
    assert(result);

    glBindVertexArray(dbc.vao);

    result = SetShaderParameter("usingDiffuseMap", false);
    assert(result);

    if (dbc.material) {
        Color color = dbc.material->GetBaseColor();
        if (color.ValueMap) {
            result = SetShaderParameter("diffuseMap", m_TextureIndex[color.ValueMap->GetName()]);
            assert(result);
            // set this to tell shader to use texture
            result = SetShaderParameter("usingDiffuseMap", true);
            assert(result);
        }
        else
        {
            result = SetShaderParameter("diffuseColor", Vector3f({color.Value[0], color.Value[1], color.Value[2]}));
            assert(result);
        }

        color = dbc.material->GetSpecularColor();
        result = SetShaderParameter("specularColor", Vector3f({color.Value[0], color.Value[1], color.Value[2]}));
        assert(result);

        Parameter param = dbc.material->GetSpecularPower();
        result = SetShaderParameter("specularPower", param.Value);
        assert(result);
    }

    glDrawElements(dbc.mode, dbc.count, dbc.type, 0x00);
}

void OpenGLGraphicsManagerCommonBase::DrawBatchDepthOnly(const DrawBatchContext& context)
{
    const OpenGLDrawBatchContext& dbc = dynamic_cast<const OpenGLDrawBatchContext&>(context);

    bool result = SetShaderParameter("modelMatrix", dbc.trans);
    assert(result);

    glBindVertexArray(dbc.vao);

    glDrawElements(dbc.mode, dbc.count, dbc.type, 0x00);
}

intptr_t OpenGLGraphicsManagerCommonBase::GenerateShadowMap(const Light& light)
{
    // Depth texture. Slower than a depth buffer, but you can sample it later in your shader
    GLuint depthTexture;
    glGenTextures(1, &depthTexture);

    // register the shadow map
    return static_cast<intptr_t>(depthTexture);
}

void OpenGLGraphicsManagerCommonBase::BeginShadowMap(const Light& light, const intptr_t shadowmap)
{
    const int32_t kShadowMapWidth = 1024;
    const int32_t kShadowMapHeight = 1024;

    // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
    glGenFramebuffers(1, &m_ShadowMapFramebufferName);

    glBindFramebuffer(GL_FRAMEBUFFER, m_ShadowMapFramebufferName);

    GLuint depthTexture = (GLuint) shadowmap;
    glActiveTexture(GL_TEXTURE0 + depthTexture);
    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, kShadowMapWidth, kShadowMapHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);

#ifdef OPENGL_ES
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);
#else
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);
#endif

    // Always check that our framebuffer is ok
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        assert(0);

    glDrawBuffers(0, nullptr); // No color buffer is drawn to.
    glDepthMask(GL_TRUE);
    glClear(GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, kShadowMapWidth, kShadowMapHeight);

#if 1
    // now set per frame constant
    Matrix4X4f view;
    Matrix4X4f projection;
    Vector3f position;
    memcpy(&position, &light.m_lightPosition, sizeof position); 
    Vector4f tmp = light.m_lightPosition + light.m_lightDirection;
    Vector3f lookAt; 
    memcpy(&lookAt, &tmp, sizeof lookAt);
    Vector3f up = { 0.0f, 0.0f, 1.0f };
    BuildViewRHMatrix(view, position, lookAt, up);

    float fieldOfView = PI / 3.0f;
    float nearClipDistance = 1.0f;
    float farClipDistance = 100.0f;
    float screenAspect = 1.0f;

    // Build the perspective projection matrix.
    BuildPerspectiveFovRHMatrix(projection, fieldOfView, screenAspect, nearClipDistance, farClipDistance);

    Matrix4X4f depthVP = view * projection;
#else
    DrawFrameContext& frameContext = m_Frames[m_nFrameIndex].frameContext;

    Matrix4X4f depthVP = frameContext.m_viewMatrix * frameContext.m_projectionMatrix;
#endif

    SetShaderParameter("depthVP", depthVP);
}

void OpenGLGraphicsManagerCommonBase::EndShadowMap(const intptr_t shadowmap)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDeleteFramebuffers(1, &m_ShadowMapFramebufferName);

    const GfxConfiguration& conf = g_pApp->GetConfiguration();
    glViewport(0, 0, conf.screenWidth, conf.screenHeight);
}

#ifdef DEBUG

void OpenGLGraphicsManagerCommonBase::DrawPoint(const Point &point, const Vector3f& color)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Point), point, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_POINTS;
    dbc.count   = 1;
    dbc.color   = color;
    BuildIdentityMatrix(dbc.trans);

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLGraphicsManagerCommonBase::DrawPoints(const Point* buffer, const size_t count, const Matrix4X4f& trans, const Vector3f& color)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Point) * count, buffer, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_POINTS;
    dbc.count   = static_cast<GLsizei>(count);
    dbc.color   = color;
    dbc.trans   = trans;

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLGraphicsManagerCommonBase::DrawPointSet(const PointSet& point_set, const Vector3f& color)
{
    Matrix4X4f trans;
    BuildIdentityMatrix(trans);

    DrawPointSet(point_set, trans, color);
}

void OpenGLGraphicsManagerCommonBase::DrawPointSet(const PointSet& point_set, const Matrix4X4f& trans, const Vector3f& color)
{
    auto count = point_set.size();
    Point* buffer = new Point[count];
    int i = 0;
    for(auto point_ptr : point_set)
    {
        buffer[i++] = *point_ptr;
    }

    DrawPoints(buffer, count, trans, color);

    delete[] buffer;
}

void OpenGLGraphicsManagerCommonBase::DrawLine(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color)
{
    auto count = vertices.size();
    GLfloat* _vertices = new GLfloat[3 * count];

    for (auto i = 0; i < count; i++)
    {
        _vertices[3 * i] = vertices[i]->data[0];
        _vertices[3 * i + 1] = vertices[i]->data[1];
        _vertices[3 * i + 2] = vertices[i]->data[2];
    }

    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * count, _vertices, GL_STATIC_DRAW);

    delete[] _vertices;

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_LINES;
    dbc.count   = static_cast<GLsizei>(count);
    dbc.color   = color;
    dbc.trans   = trans;

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLGraphicsManagerCommonBase::DrawLine(const PointList& vertices, const Vector3f& color)
{
    Matrix4X4f trans;
    BuildIdentityMatrix(trans);

    DrawLine(vertices, trans, color);
}

void OpenGLGraphicsManagerCommonBase::DrawLine(const Point& from, const Point& to, const Vector3f& color)
{
    PointList point_list;
    point_list.push_back(make_shared<Point>(from));
    point_list.push_back(make_shared<Point>(to));

    DrawLine(point_list, color);
}

void OpenGLGraphicsManagerCommonBase::DrawTriangle(const PointList& vertices, const Vector3f& color)
{
    Matrix4X4f trans;
    BuildIdentityMatrix(trans);

    DrawTriangle(vertices, trans, color);
}

void OpenGLGraphicsManagerCommonBase::DrawTriangle(const PointList& vertices, const Matrix4X4f& trans, const Vector3f& color)
{
    auto count = vertices.size();
    assert(count >= 3);

    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    Vector3f* data = new Vector3f[count];
    for(auto i = 0; i < count; i++)
    {
        data[i] = *vertices[i];
    }
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vector3f) * count, data, GL_STATIC_DRAW);
    delete[] data;

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_TRIANGLES;
    dbc.count   = static_cast<GLsizei>(vertices.size());
    dbc.color   = color;
    dbc.trans   = trans;

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLGraphicsManagerCommonBase::DrawTriangleStrip(const PointList& vertices, const Vector3f& color)
{
    auto count = vertices.size();
    assert(count >= 3);

    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLuint buffer_id;

    // Generate an ID for the vertex buffer.
    glGenBuffers(1, &buffer_id);

    // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    Vector3f* data = new Vector3f[count];
    for(auto i = 0; i < count; i++)
    {
        data[i] = *vertices[i];
    }
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vector3f) * count, data, GL_STATIC_DRAW);
    delete[] data;

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    m_DebugBuffers.push_back(buffer_id);

    DebugDrawBatchContext& dbc = *(new DebugDrawBatchContext);
    dbc.vao     = vao;
    dbc.mode    = GL_TRIANGLE_STRIP;
    dbc.count   = static_cast<GLsizei>(vertices.size());
    dbc.color   = color * 0.5f;

    m_DebugDrawBatchContext.push_back(std::move(dbc));
}

void OpenGLGraphicsManagerCommonBase::ClearDebugBuffers()
{
    for (auto dbc : m_DebugDrawBatchContext) {
        glDeleteVertexArrays(1, &dbc.vao);
    }

    m_DebugDrawBatchContext.clear();

    for (auto buf : m_DebugBuffers) {
        glDeleteBuffers(1, &buf);
    }

    m_DebugBuffers.clear();
}

void OpenGLGraphicsManagerCommonBase::RenderDebugBuffers()
{
    auto debugShaderProgram = g_pShaderManager->GetDefaultShaderProgram(DefaultShaderIndex::Debug);

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    UseShaderProgram(debugShaderProgram);

    SetPerFrameShaderParameters(m_Frames[m_nFrameIndex].frameContext);

    for (auto dbc : m_DebugDrawBatchContext)
    {
        SetShaderParameter("FrontColor", dbc.color);
        SetShaderParameter("modelMatrix", dbc.trans);

        glBindVertexArray(dbc.vao);
        glDrawArrays(dbc.mode, 0x00, dbc.count);
    }
}

void OpenGLGraphicsManagerCommonBase::DrawOverlay(const intptr_t shadowmap, float vp_left, float vp_top, float vp_width, float vp_height)
{
    GLuint vao;
    glGenVertexArrays(1, &vao);

    // Bind the vertex array object to store all the buffers and vertex attributes we create here.
    glBindVertexArray(vao);

    GLint texture_id = (GLuint) shadowmap;

    glActiveTexture(GL_TEXTURE0 + texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    auto result = SetShaderParameter("depthSampler", texture_id);
    assert(result);

    GLfloat vertices[] = {
        vp_left, vp_top, 0.0f,
        vp_left, vp_top - vp_height, 0.0f,
        vp_left + vp_width, vp_top, 0.0f,
        vp_left + vp_width, vp_top - vp_height, 0.0f
    };

    GLfloat uv[] = {
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 0.0f
    };

    GLuint buffer_id[2];

    // Generate an ID for the vertex buffer.
    glGenBuffers(2, buffer_id);

    // Bind the vertex buffer and load the vertex (position) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);

    glVertexAttribPointer(0, 3, GL_FLOAT, false, 0, 0);

    // Bind the vertex buffer and load the vertex (uv) data into the vertex buffer.
    glBindBuffer(GL_ARRAY_BUFFER, buffer_id[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(uv), uv, GL_STATIC_DRAW);

    glEnableVertexAttribArray(1);

    glVertexAttribPointer(1, 2, GL_FLOAT, false, 0, 0);

    glDrawArrays(GL_TRIANGLE_STRIP, 0x00, 4);

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(2, buffer_id);
}

#endif