// this is a common code snip file
// should be used by include into source code
// do not compile it seperately

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

bool OpenGLGraphicsManagerCommonBase::SetPerFrameShaderParameters(GLuint shader)
{
    unsigned int location;

    // Set the world matrix in the vertex shader.
    location = glGetUniformLocation(shader, "worldMatrix");
    if(location == -1)
    {
            return false;
    }
    glUniformMatrix4fv(location, 1, false, m_DrawFrameContext.m_worldMatrix);

    // Set the view matrix in the vertex shader.
    location = glGetUniformLocation(shader, "viewMatrix");
    if(location == -1)
    {
            return false;
    }
    glUniformMatrix4fv(location, 1, false, m_DrawFrameContext.m_viewMatrix);

    // Set the projection matrix in the vertex shader.
    location = glGetUniformLocation(shader, "projectionMatrix");
    if(location == -1)
    {
            return false;
    }
    glUniformMatrix4fv(location, 1, false, m_DrawFrameContext.m_projectionMatrix);

    location = glGetUniformLocation(shader, "ambientColor");
    if(location == -1)
    {
            return false;
    }
    glUniform4fv(location, 1, m_DrawFrameContext.m_ambientColor);

    location = glGetUniformLocation(shader, "numLights");
    if(location == -1)
    {
            return false;
    }
    glUniform1i(location, m_DrawFrameContext.m_lights.size());

    // Set lighting parameters for PS shader
    for (size_t i = 0; i < m_DrawFrameContext.m_lights.size(); i++)
    {
        ostringstream ss;
        string uniformName;

        ss << "allLights[" << i << "]." << "lightPosition" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(shader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform4fv(location, 1, m_DrawFrameContext.m_lights[i].m_lightPosition);

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightColor" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(shader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform4fv(location, 1, m_DrawFrameContext.m_lights[i].m_lightColor);

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightIntensity" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(shader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform1f(location, m_DrawFrameContext.m_lights[i].m_lightIntensity);

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightDirection" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(shader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform3fv(location, 1, m_DrawFrameContext.m_lights[i].m_lightDirection);

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightDistAttenCurveType" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(shader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform1i(location, (GLint)m_DrawFrameContext.m_lights[i].m_lightDistAttenCurveType);

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightDistAttenCurveParams" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(shader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform1fv(location, 5, m_DrawFrameContext.m_lights[i].m_lightDistAttenCurveParams);

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightAngleAttenCurveType" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(shader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform1i(location, (GLint)m_DrawFrameContext.m_lights[i].m_lightAngleAttenCurveType);

        ss.clear();
        ss.seekp(0);
        ss << "allLights[" << i << "]." << "lightAngleAttenCurveParams" << ends;
        uniformName = ss.str();
        location = glGetUniformLocation(shader, uniformName.c_str());
        if(location == -1)
        {
                return false;
        }
        glUniform1fv(location, 5, m_DrawFrameContext.m_lights[i].m_lightAngleAttenCurveParams);

    }

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const Matrix4X4f& param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniformMatrix4fv(location, 1, false, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const Vector3f& param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform3fv(location, 1, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const float param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform1f(location, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const int param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
    if(location == -1)
    {
            return false;
    }
    glUniform1i(location, param);

    return true;
}

bool OpenGLGraphicsManagerCommonBase::SetPerBatchShaderParameters(GLuint shader, const char* paramName, const bool param)
{
    unsigned int location;

    location = glGetUniformLocation(shader, paramName);
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

                DrawBatchContext& dbc = *(new DrawBatchContext);
                dbc.vao     = vao;
                dbc.mode    = mode;
                dbc.type    = type;
                dbc.count   = indexCount;
                dbc.node    = pGeometryNode;
                dbc.material = material;
                m_DrawBatchContext.push_back(std::move(dbc));
            }
        }
    }

    return;
}

void OpenGLGraphicsManagerCommonBase::ClearBuffers()
{
    for (auto dbc : m_DrawBatchContext) {
        glDeleteVertexArrays(1, &dbc.vao);
    }

    m_DrawBatchContext.clear();

    for (auto buf : m_Buffers) {
        glDeleteBuffers(1, &buf);
    }

    for (auto texture : m_Textures) {
        glDeleteTextures(1, &texture);
    }

    m_Buffers.clear();
    m_Textures.clear();

}

void OpenGLGraphicsManagerCommonBase::RenderBuffers()
{
    bool result;

    GLuint shaderProgram = *reinterpret_cast<GLuint*>(g_pShaderManager->GetDefaultShaderProgram());

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    glUseProgram(shaderProgram);

    result = SetPerFrameShaderParameters(shaderProgram);
    assert(result);

    for (auto dbc : m_DrawBatchContext)
    {
        Matrix4X4f trans;
        if (void* rigidBody = dbc.node->RigidBody()) {
            // the geometry has rigid body bounded, we blend the simlation result here.
            Matrix4X4f simulated_result = g_pPhysicsManager->GetRigidBodyTransform(rigidBody);

            BuildIdentityMatrix(trans);

            // apply the rotation part of the simlation result
            memcpy(trans[0], simulated_result[0], sizeof(float) * 3);
            memcpy(trans[1], simulated_result[1], sizeof(float) * 3);
            memcpy(trans[2], simulated_result[2], sizeof(float) * 3);

            // replace the translation part of the matrix with simlation result directly
            memcpy(trans[3], simulated_result[3], sizeof(float) * 3);

        } else {
            trans = *dbc.node->GetCalculatedTransform();
        }

        result = SetPerBatchShaderParameters(shaderProgram, "modelMatrix", trans);
        assert(result);

        glBindVertexArray(dbc.vao);

        /* well, we have different material for each index buffer so we can not draw them together
        * in future we should group indicies according to its material and draw them together
        auto indexBufferCount = dbc.counts.size();
        const GLvoid ** pIndicies = new const GLvoid*[indexBufferCount];
        memset(pIndicies, 0x00, sizeof(GLvoid*) * indexBufferCount);
        // Render the vertex buffer using the index buffer.
        glMultiDrawElements(dbc.mode, dbc.counts.data(), dbc.type, pIndicies, indexBufferCount);
        delete[] pIndicies;
        */

        result = SetPerBatchShaderParameters(shaderProgram, "usingDiffuseMap", false);
        assert(result);

#if 0
        result = SetPerBatchShaderParameters(shaderProgram, "usingNormalMap", false);
        assert(result);
#endif

        if (dbc.material) {
            Color color = dbc.material->GetBaseColor();
            if (color.ValueMap) {
                result = SetPerBatchShaderParameters(shaderProgram, "diffuseMap", m_TextureIndex[color.ValueMap->GetName()]);
                assert(result);
                // set this to tell shader to use texture
                result = SetPerBatchShaderParameters(shaderProgram, "usingDiffuseMap", true);
                assert(result);
            }
            else
            {
                result = SetPerBatchShaderParameters(shaderProgram, "diffuseColor", Vector3f({color.Value[0], color.Value[1], color.Value[2]}));
                assert(result);
            }

#if 0
            Normal normal = dbc.material->GetNormal();
            if (normal.ValueMap) {
                result = SetPerBatchShaderParameters(shaderProgram, "normalMap", m_TextureIndex[normal.ValueMap->GetName()]);
                assert(result);
                // set this to tell shader to use texture
                result = SetPerBatchShaderParameters(shaderProgram, "usingNormalMap", true);
                assert(result);
            }
#endif

            color = dbc.material->GetSpecularColor();
            result = SetPerBatchShaderParameters(shaderProgram, "specularColor", Vector3f({color.Value[0], color.Value[1], color.Value[2]}));
            assert(result);

            Parameter param = dbc.material->GetSpecularPower();
            result = SetPerBatchShaderParameters(shaderProgram, "specularPower", param.Value);
            assert(result);
        }

        glDrawElements(dbc.mode, dbc.count, dbc.type, 0x00);
    }

    return;
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
    GLuint debugShaderProgram = *reinterpret_cast<GLuint*>(g_pShaderManager->GetDebugShaderProgram());

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    glUseProgram(debugShaderProgram);

    SetPerFrameShaderParameters(debugShaderProgram);

    for (auto dbc : m_DebugDrawBatchContext)
    {
        SetPerBatchShaderParameters(debugShaderProgram, "FrontColor", dbc.color);
        SetPerBatchShaderParameters(debugShaderProgram, "modelMatrix", dbc.trans);

        glBindVertexArray(dbc.vao);
        glDrawArrays(dbc.mode, 0x00, dbc.count);
    }
}

#endif