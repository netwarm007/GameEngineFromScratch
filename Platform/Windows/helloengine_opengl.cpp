// include the basic windows header file
#include <windows.h>
#include <windowsx.h>
#include <tchar.h>
#include <GL/gl.h>
#include <fstream>

#include "vectormath.h"

using namespace std;

/////////////
// DEFINES //
/////////////
#define WGL_DRAW_TO_WINDOW_ARB         0x2001
#define WGL_ACCELERATION_ARB           0x2003
#define WGL_SWAP_METHOD_ARB            0x2007
#define WGL_SUPPORT_OPENGL_ARB         0x2010
#define WGL_DOUBLE_BUFFER_ARB          0x2011
#define WGL_PIXEL_TYPE_ARB             0x2013
#define WGL_COLOR_BITS_ARB             0x2014
#define WGL_DEPTH_BITS_ARB             0x2022
#define WGL_STENCIL_BITS_ARB           0x2023
#define WGL_FULL_ACCELERATION_ARB      0x2027
#define WGL_SWAP_EXCHANGE_ARB          0x2028
#define WGL_TYPE_RGBA_ARB              0x202B
#define WGL_CONTEXT_MAJOR_VERSION_ARB  0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB  0x2092
#define GL_ARRAY_BUFFER                   0x8892
#define GL_STATIC_DRAW                    0x88E4
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_VERTEX_SHADER                  0x8B31
#define GL_COMPILE_STATUS                 0x8B81
#define GL_LINK_STATUS                    0x8B82
#define GL_INFO_LOG_LENGTH                0x8B84
#define GL_TEXTURE0                       0x84C0
#define GL_BGRA                           0x80E1
#define GL_ELEMENT_ARRAY_BUFFER           0x8893

//////////////
// TYPEDEFS //
//////////////
typedef BOOL (WINAPI   * PFNWGLCHOOSEPIXELFORMATARBPROC) (HDC hdc, const int *piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
typedef HGLRC (WINAPI  * PFNWGLCREATECONTEXTATTRIBSARBPROC) (HDC hDC, HGLRC hShareContext, const int *attribList);
typedef BOOL (WINAPI   * PFNWGLSWAPINTERVALEXTPROC) (int interval);
typedef void (APIENTRY * PFNGLATTACHSHADERPROC) (GLuint program, GLuint shader);
typedef void (APIENTRY * PFNGLBINDBUFFERPROC) (GLenum target, GLuint buffer);
typedef void (APIENTRY * PFNGLBINDVERTEXARRAYPROC) (GLuint array);
typedef void (APIENTRY * PFNGLBUFFERDATAPROC) (GLenum target, ptrdiff_t size, const GLvoid *data, GLenum usage);
typedef void (APIENTRY * PFNGLCOMPILESHADERPROC) (GLuint shader);
typedef GLuint(APIENTRY * PFNGLCREATEPROGRAMPROC) (void);
typedef GLuint(APIENTRY * PFNGLCREATESHADERPROC) (GLenum type);
typedef void (APIENTRY * PFNGLDELETEBUFFERSPROC) (GLsizei n, const GLuint *buffers);
typedef void (APIENTRY * PFNGLDELETEPROGRAMPROC) (GLuint program);
typedef void (APIENTRY * PFNGLDELETESHADERPROC) (GLuint shader);
typedef void (APIENTRY * PFNGLDELETEVERTEXARRAYSPROC) (GLsizei n, const GLuint *arrays);
typedef void (APIENTRY * PFNGLDETACHSHADERPROC) (GLuint program, GLuint shader);
typedef void (APIENTRY * PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint index);
typedef void (APIENTRY * PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *buffers);
typedef void (APIENTRY * PFNGLGENVERTEXARRAYSPROC) (GLsizei n, GLuint *arrays);
typedef GLint(APIENTRY * PFNGLGETATTRIBLOCATIONPROC) (GLuint program, const char *name);
typedef void (APIENTRY * PFNGLGETPROGRAMINFOLOGPROC) (GLuint program, GLsizei bufSize, GLsizei *length, char *infoLog);
typedef void (APIENTRY * PFNGLGETPROGRAMIVPROC) (GLuint program, GLenum pname, GLint *params);
typedef void (APIENTRY * PFNGLGETSHADERINFOLOGPROC) (GLuint shader, GLsizei bufSize, GLsizei *length, char *infoLog);
typedef void (APIENTRY * PFNGLGETSHADERIVPROC) (GLuint shader, GLenum pname, GLint *params);
typedef void (APIENTRY * PFNGLLINKPROGRAMPROC) (GLuint program);
typedef void (APIENTRY * PFNGLSHADERSOURCEPROC) (GLuint shader, GLsizei count, const char* *string, const GLint *length);
typedef void (APIENTRY * PFNGLUSEPROGRAMPROC) (GLuint program);
typedef void (APIENTRY * PFNGLVERTEXATTRIBPOINTERPROC) (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid *pointer);
typedef void (APIENTRY * PFNGLBINDATTRIBLOCATIONPROC) (GLuint program, GLuint index, const char *name);
typedef GLint(APIENTRY * PFNGLGETUNIFORMLOCATIONPROC) (GLuint program, const char *name);
typedef void (APIENTRY * PFNGLUNIFORMMATRIX4FVPROC) (GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
typedef void (APIENTRY * PFNGLACTIVETEXTUREPROC) (GLenum texture);
typedef void (APIENTRY * PFNGLUNIFORM1IPROC) (GLint location, GLint v0);
typedef void (APIENTRY * PFNGLGENERATEMIPMAPPROC) (GLenum target);
typedef void (APIENTRY * PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint index);
typedef void (APIENTRY * PFNGLUNIFORM3FVPROC) (GLint location, GLsizei count, const GLfloat *value);
typedef void (APIENTRY * PFNGLUNIFORM4FVPROC) (GLint location, GLsizei count, const GLfloat *value);

PFNGLATTACHSHADERPROC glAttachShader;
PFNGLBINDBUFFERPROC glBindBuffer;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
PFNGLBUFFERDATAPROC glBufferData;
PFNGLCOMPILESHADERPROC glCompileShader;
PFNGLCREATEPROGRAMPROC glCreateProgram;
PFNGLCREATESHADERPROC glCreateShader;
PFNGLDELETEBUFFERSPROC glDeleteBuffers;
PFNGLDELETEPROGRAMPROC glDeleteProgram;
PFNGLDELETESHADERPROC glDeleteShader;
PFNGLDELETEVERTEXARRAYSPROC glDeleteVertexArrays;
PFNGLDETACHSHADERPROC glDetachShader;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
PFNGLGENBUFFERSPROC glGenBuffers;
PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
PFNGLGETATTRIBLOCATIONPROC glGetAttribLocation;
PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog;
PFNGLGETPROGRAMIVPROC glGetProgramiv;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
PFNGLGETSHADERIVPROC glGetShaderiv;
PFNGLLINKPROGRAMPROC glLinkProgram;
PFNGLSHADERSOURCEPROC glShaderSource;
PFNGLUSEPROGRAMPROC glUseProgram;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;
PFNGLBINDATTRIBLOCATIONPROC glBindAttribLocation;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv;
PFNGLACTIVETEXTUREPROC glActiveTexture;
PFNGLUNIFORM1IPROC glUniform1i;
PFNGLGENERATEMIPMAPPROC glGenerateMipmap;
PFNGLDISABLEVERTEXATTRIBARRAYPROC glDisableVertexAttribArray;
PFNGLUNIFORM3FVPROC glUniform3fv;
PFNGLUNIFORM4FVPROC glUniform4fv;

PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB;
PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB;
PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;

typedef struct VertexType
{
    VectorType position;
    VectorType color;
} VertexType;

HDC     g_deviceContext = 0;
HGLRC   g_renderingContext = 0;
char    g_videoCardDescription[128];

const bool VSYNC_ENABLED = true;
const float SCREEN_DEPTH = 1000.0f;
const float SCREEN_NEAR = 0.1f;

int     g_vertexCount, g_indexCount;
unsigned int g_vertexArrayId, g_vertexBufferId, g_indexBufferId;

unsigned int g_vertexShader;
unsigned int g_fragmentShader;
unsigned int g_shaderProgram;

const char VS_SHADER_SOURCE_FILE[] = "color.vs";
const char PS_SHADER_SOURCE_FILE[] = "color.ps";

float g_positionX = 0, g_positionY = 0, g_positionZ = -10;
float g_rotationX = 0, g_rotationY = 0, g_rotationZ = 0;
float g_worldMatrix[16];
float g_viewMatrix[16];
float g_projectionMatrix[16];

bool InitializeOpenGL(HWND hwnd, int screenWidth, int screenHeight, float screenDepth, float screenNear, bool vsync)
{
        int attributeListInt[19];
        int pixelFormat[1];
        unsigned int formatCount;
        int result;
        PIXELFORMATDESCRIPTOR pixelFormatDescriptor;
        int attributeList[5];
        float fieldOfView, screenAspect;
        char *vendorString, *rendererString;


        // Get the device context for this window.
        g_deviceContext = GetDC(hwnd);
        if(!g_deviceContext)
        {
                return false;
        }

        // Support for OpenGL rendering.
        attributeListInt[0] = WGL_SUPPORT_OPENGL_ARB;
        attributeListInt[1] = TRUE;

        // Support for rendering to a window.
        attributeListInt[2] = WGL_DRAW_TO_WINDOW_ARB;
        attributeListInt[3] = TRUE;

        // Support for hardware acceleration.
        attributeListInt[4] = WGL_ACCELERATION_ARB;
        attributeListInt[5] = WGL_FULL_ACCELERATION_ARB;

        // Support for 24bit color.
        attributeListInt[6] = WGL_COLOR_BITS_ARB;
        attributeListInt[7] = 24;

        // Support for 24 bit depth buffer.
        attributeListInt[8] = WGL_DEPTH_BITS_ARB;
        attributeListInt[9] = 24;

        // Support for double buffer.
        attributeListInt[10] = WGL_DOUBLE_BUFFER_ARB;
        attributeListInt[11] = TRUE;

        // Support for swapping front and back buffer.
        attributeListInt[12] = WGL_SWAP_METHOD_ARB;
        attributeListInt[13] = WGL_SWAP_EXCHANGE_ARB;

        // Support for the RGBA pixel type.
        attributeListInt[14] = WGL_PIXEL_TYPE_ARB;
        attributeListInt[15] = WGL_TYPE_RGBA_ARB;

        // Support for a 8 bit stencil buffer.
        attributeListInt[16] = WGL_STENCIL_BITS_ARB;
        attributeListInt[17] = 8;

        // Null terminate the attribute list.
        attributeListInt[18] = 0;

        
        // Query for a pixel format that fits the attributes we want.
        result = wglChoosePixelFormatARB(g_deviceContext, attributeListInt, NULL, 1, pixelFormat, &formatCount);
        if(result != 1)
        {
                return false;
        }

        // If the video card/display can handle our desired pixel format then we set it as the current one.
        result = SetPixelFormat(g_deviceContext, pixelFormat[0], &pixelFormatDescriptor);
        if(result != 1)
        {
                return false;
        }

        // Set the 4.0 version of OpenGL in the attribute list.
        attributeList[0] = WGL_CONTEXT_MAJOR_VERSION_ARB;
        attributeList[1] = 4;
        attributeList[2] = WGL_CONTEXT_MINOR_VERSION_ARB;
        attributeList[3] = 0;

        // Null terminate the attribute list.
        attributeList[4] = 0;

        // Create a OpenGL 4.0 rendering context.
        g_renderingContext = wglCreateContextAttribsARB(g_deviceContext, 0, attributeList);
        if(g_renderingContext == NULL)
        {
                return false;
        }

        // Set the rendering context to active.
        result = wglMakeCurrent(g_deviceContext, g_renderingContext);
        if(result != 1)
        {
                return false;
        }

        // Set the depth buffer to be entirely cleared to 1.0 values.
        glClearDepth(1.0f);

        // Enable depth testing.
        glEnable(GL_DEPTH_TEST);

        // Set the polygon winding to front facing for the left handed system.
        glFrontFace(GL_CW);

        // Enable back face culling.
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        // Initialize the world/model matrix to the identity matrix.
        BuildIdentityMatrix(g_worldMatrix);

        // Set the field of view and screen aspect ratio.
        fieldOfView = PI / 4.0f;
        screenAspect = (float)screenWidth / (float)screenHeight;

        // Build the perspective projection matrix.
        BuildPerspectiveFovLHMatrix(g_projectionMatrix, fieldOfView, screenAspect, screenNear, screenDepth);

        // Get the name of the video card.
        vendorString = (char*)glGetString(GL_VENDOR);
        rendererString = (char*)glGetString(GL_RENDERER);
        // Store the video card name in a class member variable so it can be retrieved later.
        strcpy_s(g_videoCardDescription, vendorString);
        strcat_s(g_videoCardDescription, " - ");
        strcat_s(g_videoCardDescription, rendererString);

        // Turn on or off the vertical sync depending on the input bool value.
        if(vsync)
        {
                result = wglSwapIntervalEXT(1);
        }
        else
        {
                result = wglSwapIntervalEXT(0);
        }

        // Check if vsync was set correctly.
        if(result != 1)
        {
                return false;
        }

        return true;
}

bool LoadExtensionList()
{
        // Load the OpenGL extensions that this application will be using.
        wglChoosePixelFormatARB = (PFNWGLCHOOSEPIXELFORMATARBPROC)wglGetProcAddress("wglChoosePixelFormatARB");
        if(!wglChoosePixelFormatARB)
        {
                return false;
        }

        wglCreateContextAttribsARB = (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");
        if(!wglCreateContextAttribsARB)
        {
                return false;
        }

        wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
        if(!wglSwapIntervalEXT)
        {
                return false;
        }

        glAttachShader = (PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader");
        if(!glAttachShader)
        {
                return false;
        }

        glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
        if(!glBindBuffer)
        {
                return false;
        }

        glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)wglGetProcAddress("glBindVertexArray");
        if(!glBindVertexArray)
        {
                return false;
        }

        glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
        if(!glBufferData)
        {
                return false;
        }

        glCompileShader = (PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader");
        if(!glCompileShader)
        {
                return false;
        }

        glCreateProgram = (PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram");
        if(!glCreateProgram)
        {
                return false;
        }

        glCreateShader = (PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader");
        if(!glCreateShader)
        {
                return false;
        }

        glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)wglGetProcAddress("glDeleteBuffers");
        if(!glDeleteBuffers)
        {
                return false;
        }
        
        glDeleteProgram = (PFNGLDELETEPROGRAMPROC)wglGetProcAddress("glDeleteProgram");
        if(!glDeleteProgram)
        {
                return false;
        }

        glDeleteShader = (PFNGLDELETESHADERPROC)wglGetProcAddress("glDeleteShader");
        if(!glDeleteShader)
        {
                return false;
        }

        glDeleteVertexArrays = (PFNGLDELETEVERTEXARRAYSPROC)wglGetProcAddress("glDeleteVertexArrays");
        if(!glDeleteVertexArrays)
        {
                return false;
        }

        glDetachShader = (PFNGLDETACHSHADERPROC)wglGetProcAddress("glDetachShader");
        if(!glDetachShader)
        {
                return false;
        }

        glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray");
        if(!glEnableVertexAttribArray)
        {
                return false;
        }

        glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
        if(!glGenBuffers)
        {
                return false;
        }

        glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)wglGetProcAddress("glGenVertexArrays");
        if(!glGenVertexArrays)
        {
                return false;
        }

        glGetAttribLocation = (PFNGLGETATTRIBLOCATIONPROC)wglGetProcAddress("glGetAttribLocation");
        if(!glGetAttribLocation)
        {
                return false;
        }

        glGetProgramInfoLog = (PFNGLGETPROGRAMINFOLOGPROC)wglGetProcAddress("glGetProgramInfoLog");
        if(!glGetProgramInfoLog)
        {
                return false;
        }

        glGetProgramiv = (PFNGLGETPROGRAMIVPROC)wglGetProcAddress("glGetProgramiv");
        if(!glGetProgramiv)
        {
                return false;
        }

        glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)wglGetProcAddress("glGetShaderInfoLog");
        if(!glGetShaderInfoLog)
        {
                return false;
        }

        glGetShaderiv = (PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv");
        if(!glGetShaderiv)
        {
                return false;
        }

        glLinkProgram = (PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram");
        if(!glLinkProgram)
        {
                return false;
        }

        glShaderSource = (PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource");
        if(!glShaderSource)
        {
                return false;
        }

        glUseProgram = (PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram");
        if(!glUseProgram)
        {
                return false;
        }

        glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer");
        if(!glVertexAttribPointer)
        {
                return false;
        }

        glBindAttribLocation = (PFNGLBINDATTRIBLOCATIONPROC)wglGetProcAddress("glBindAttribLocation");
        if(!glBindAttribLocation)
        {
                return false;
        }

        glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation");
        if(!glGetUniformLocation)
        {
                return false;
        }

        glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)wglGetProcAddress("glUniformMatrix4fv");
        if(!glUniformMatrix4fv)
        {
                return false;
        }

        glActiveTexture = (PFNGLACTIVETEXTUREPROC)wglGetProcAddress("glActiveTexture");
        if(!glActiveTexture)
        {
                return false;
        }

        glUniform1i = (PFNGLUNIFORM1IPROC)wglGetProcAddress("glUniform1i");
        if(!glUniform1i)
        {
                return false;
        }

        glGenerateMipmap = (PFNGLGENERATEMIPMAPPROC)wglGetProcAddress("glGenerateMipmap");
        if(!glGenerateMipmap)
        {
                return false;
        }

        glDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glDisableVertexAttribArray");
        if(!glDisableVertexAttribArray)
        {
                return false;
        }

        glUniform3fv = (PFNGLUNIFORM3FVPROC)wglGetProcAddress("glUniform3fv");
        if(!glUniform3fv)
        {
                return false;
        }

        glUniform4fv = (PFNGLUNIFORM4FVPROC)wglGetProcAddress("glUniform4fv");
        if(!glUniform4fv)
        {
                return false;
        }

        return true;
}

void FinalizeOpenGL(HWND hwnd)
{
        // Release the rendering context.
        if(g_renderingContext)
        {
                wglMakeCurrent(NULL, NULL);
                wglDeleteContext(g_renderingContext);
                g_renderingContext = 0;
        }

        // Release the device context.
        if(g_deviceContext)
        {
                ReleaseDC(hwnd, g_deviceContext);
                g_deviceContext = 0;
        }
}

void GetVideoCardInfo(char* cardName)
{
        strcpy_s(cardName, 128, g_videoCardDescription);
        return;
}

bool InitializeExtensions(HWND hwnd)
{
        HDC deviceContext;
        PIXELFORMATDESCRIPTOR pixelFormat;
        int error;
        HGLRC renderContext;
        bool result;


        // Get the device context for this window.
        deviceContext = GetDC(hwnd);
        if(!deviceContext)
        {
                return false;
        }

        // Set a temporary default pixel format.
        error = SetPixelFormat(deviceContext, 1, &pixelFormat);
        if(error != 1)
        {
                return false;
        }

        // Create a temporary rendering context.
        renderContext = wglCreateContext(deviceContext);
        if(!renderContext)
        {
                return false;
        }

        // Set the temporary rendering context as the current rendering context for this window.
        error = wglMakeCurrent(deviceContext, renderContext);
        if(error != 1)
        {
                return false;
        }

        // Initialize the OpenGL extensions needed for this application.  Note that a temporary rendering context was needed to do so.
        result = LoadExtensionList();
        if(!result)
        {
                return false;
        }

        // Release the temporary rendering context now that the extensions have been loaded.
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(renderContext);
        renderContext = NULL;

        // Release the device context for this window.
        ReleaseDC(hwnd, deviceContext);
        deviceContext = 0;

        return true;
}

void OutputShaderErrorMessage(HWND hwnd, unsigned int shaderId, const char* shaderFilename)
{
        int logSize, i;
        char* infoLog;
        ofstream fout;
        wchar_t newString[128];
        unsigned int error;
        size_t convertedChars;


        // Get the size of the string containing the information log for the failed shader compilation message.
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &logSize);

        // Increment the size by one to handle also the null terminator.
        logSize++;

        // Create a char buffer to hold the info log.
        infoLog = new char[logSize];
        if(!infoLog)
        {
                return;
        }

        // Now retrieve the info log.
        glGetShaderInfoLog(shaderId, logSize, NULL, infoLog);

        // Open a file to write the error message to.
        fout.open("shader-error.txt");

        // Write out the error message.
        for(i=0; i<logSize; i++)
        {
                fout << infoLog[i];
        }

        // Close the file.
        fout.close();

        // Convert the shader filename to a wide character string.
        error = mbstowcs_s(&convertedChars, newString, 128, shaderFilename, 128);
        if(error != 0)
        {
                return;
        }

        // Pop a message up on the screen to notify the user to check the text file for compile errors.
        MessageBoxW(hwnd, L"Error compiling shader.  Check shader-error.txt for message.", newString, MB_OK);

        return;
}

void OutputLinkerErrorMessage(HWND hwnd, unsigned int programId)
{
        int logSize, i;
        char* infoLog;
        ofstream fout;


        // Get the size of the string containing the information log for the failed shader compilation message.
        glGetProgramiv(programId, GL_INFO_LOG_LENGTH, &logSize);

        // Increment the size by one to handle also the null terminator.
        logSize++;

        // Create a char buffer to hold the info log.
        infoLog = new char[logSize];
        if(!infoLog)
        {
                return;
        }

        // Now retrieve the info log.
        glGetProgramInfoLog(programId, logSize, NULL, infoLog);

        // Open a file to write the error message to.
        fout.open("linker-error.txt");

        // Write out the error message.
        for(i=0; i<logSize; i++)
        {
                fout << infoLog[i];
        }

        // Close the file.
        fout.close();

        // Pop a message up on the screen to notify the user to check the text file for linker errors.
        MessageBox(hwnd, _T("Error compiling linker.  Check linker-error.txt for message."), _T("Linker Error"), MB_OK);
}

char* LoadShaderSourceFile(const char* filename)
{
        ifstream fin;
        int fileSize;
        char input;
        char* buffer;


        // Open the shader source file.
        fin.open(filename);

        // If it could not open the file then exit.
        if(fin.fail())
        {
                return 0;
        }

        // Initialize the size of the file.
        fileSize = 0;

        // Read the first element of the file.
        fin.get(input);

        // Count the number of elements in the text file.
        while(!fin.eof())
        {
                fileSize++;
                fin.get(input);
        }

        // Close the file for now.
        fin.close();

        // Initialize the buffer to read the shader source file into.
        buffer = new char[fileSize+1];
        if(!buffer)
        {
                return 0;
        }

        // Open the shader source file again.
        fin.open(filename);

        // Read the shader text file into the buffer as a block.
        fin.read(buffer, fileSize);

        // Close the file.
        fin.close();

        // Null terminate the buffer.
        buffer[fileSize] = '\0';

        return buffer;
}

bool InitializeShader(HWND hwnd, const char* vsFilename, const char* fsFilename)
{
        const char* vertexShaderBuffer;
        const char* fragmentShaderBuffer;
        int status;

        // Load the vertex shader source file into a text buffer.
        vertexShaderBuffer = LoadShaderSourceFile(vsFilename);
        if(!vertexShaderBuffer)
        {
                return false;
        }

        // Load the fragment shader source file into a text buffer.
        fragmentShaderBuffer = LoadShaderSourceFile(fsFilename);
        if(!fragmentShaderBuffer)
        {
                return false;
        }

        // Create a vertex and fragment shader object.
        g_vertexShader = glCreateShader(GL_VERTEX_SHADER);
        g_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

        // Copy the shader source code strings into the vertex and fragment shader objects.
        glShaderSource(g_vertexShader, 1, &vertexShaderBuffer, NULL);
        glShaderSource(g_fragmentShader, 1, &fragmentShaderBuffer, NULL);

        // Release the vertex and fragment shader buffers.
        delete [] vertexShaderBuffer;
        vertexShaderBuffer = 0;

        delete [] fragmentShaderBuffer;
        fragmentShaderBuffer = 0;

        // Compile the shaders.
        glCompileShader(g_vertexShader);
        glCompileShader(g_fragmentShader);

        // Check to see if the vertex shader compiled successfully.
        glGetShaderiv(g_vertexShader, GL_COMPILE_STATUS, &status);
        if(status != 1)
        {
                // If it did not compile then write the syntax error message out to a text file for review.
                OutputShaderErrorMessage(hwnd, g_vertexShader, vsFilename);
                return false;
        }

        // Check to see if the fragment shader compiled successfully.
        glGetShaderiv(g_fragmentShader, GL_COMPILE_STATUS, &status);
        if(status != 1)
        {
                // If it did not compile then write the syntax error message out to a text file for review.
                OutputShaderErrorMessage(hwnd, g_fragmentShader, fsFilename);
                return false;
        }

        // Create a shader program object.
        g_shaderProgram = glCreateProgram();

        // Attach the vertex and fragment shader to the program object.
        glAttachShader(g_shaderProgram, g_vertexShader);
        glAttachShader(g_shaderProgram, g_fragmentShader);

        // Bind the shader input variables.
        glBindAttribLocation(g_shaderProgram, 0, "inputPosition");
        glBindAttribLocation(g_shaderProgram, 1, "inputColor");

        // Link the shader program.
        glLinkProgram(g_shaderProgram);

        // Check the status of the link.
        glGetProgramiv(g_shaderProgram, GL_LINK_STATUS, &status);
        if(status != 1)
        {
                // If it did not link then write the syntax error message out to a text file for review.
                OutputLinkerErrorMessage(hwnd, g_shaderProgram);
                return false;
        }

        return true;
}

void ShutdownShader()
{
        // Detach the vertex and fragment shaders from the program.
        glDetachShader(g_shaderProgram, g_vertexShader);
        glDetachShader(g_shaderProgram, g_fragmentShader);

        // Delete the vertex and fragment shaders.
        glDeleteShader(g_vertexShader);
        glDeleteShader(g_fragmentShader);

        // Delete the shader program.
        glDeleteProgram(g_shaderProgram);
}

bool SetShaderParameters(float* worldMatrix, float* viewMatrix, float* projectionMatrix)
{
        unsigned int location;

        // Set the world matrix in the vertex shader.
        location = glGetUniformLocation(g_shaderProgram, "worldMatrix");
        if(location == -1)
        {
                return false;
        }
        glUniformMatrix4fv(location, 1, false, worldMatrix);

        // Set the view matrix in the vertex shader.
        location = glGetUniformLocation(g_shaderProgram, "viewMatrix");
        if(location == -1)
        {
                return false;
        }
        glUniformMatrix4fv(location, 1, false, viewMatrix);

        // Set the projection matrix in the vertex shader.
        location = glGetUniformLocation(g_shaderProgram, "projectionMatrix");
        if(location == -1)
        {
                return false;
        }
        glUniformMatrix4fv(location, 1, false, projectionMatrix);

        return true;
}

bool InitializeBuffers()
{
        VertexType vertices[] = {
            {{  1.0f,  1.0f,  1.0f }, { 1.0f, 0.0f, 0.0f }},
            {{  1.0f,  1.0f, -1.0f }, { 0.0f, 1.0f, 0.0f }},
            {{ -1.0f,  1.0f, -1.0f }, { 0.0f, 0.0f, 1.0f }},
            {{ -1.0f,  1.0f,  1.0f }, { 1.0f, 1.0f, 0.0f }},
            {{  1.0f, -1.0f,  1.0f }, { 1.0f, 0.0f, 1.0f }},
            {{  1.0f, -1.0f, -1.0f }, { 0.0f, 1.0f, 1.0f }},
            {{ -1.0f, -1.0f, -1.0f }, { 0.5f, 1.0f, 0.5f }},
            {{ -1.0f, -1.0f,  1.0f }, { 1.0f, 0.5f, 1.0f }},
        };
        uint16_t indices[] = { 1, 2, 3, 3, 2, 6, 6, 7, 3, 3, 0, 1, 0, 3, 7, 7, 6, 4, 4, 6, 5, 0, 7, 4, 1, 0, 4, 1, 4, 5, 2, 1, 5, 2, 5, 6 };

        // Set the number of vertices in the vertex array.
        g_vertexCount = sizeof(vertices) / sizeof(VertexType);

        // Set the number of indices in the index array.
        g_indexCount = sizeof(indices) / sizeof(uint16_t);

        // Allocate an OpenGL vertex array object.
        glGenVertexArrays(1, &g_vertexArrayId);

        // Bind the vertex array object to store all the buffers and vertex attributes we create here.
        glBindVertexArray(g_vertexArrayId);

        // Generate an ID for the vertex buffer.
        glGenBuffers(1, &g_vertexBufferId);

        // Bind the vertex buffer and load the vertex (position and color) data into the vertex buffer.
        glBindBuffer(GL_ARRAY_BUFFER, g_vertexBufferId);
        glBufferData(GL_ARRAY_BUFFER, g_vertexCount * sizeof(VertexType), vertices, GL_STATIC_DRAW);

        // Enable the two vertex array attributes.
        glEnableVertexAttribArray(0);  // Vertex position.
        glEnableVertexAttribArray(1);  // Vertex color.

        // Specify the location and format of the position portion of the vertex buffer.
        glBindBuffer(GL_ARRAY_BUFFER, g_vertexBufferId);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(VertexType), 0);

        // Specify the location and format of the color portion of the vertex buffer.
        glBindBuffer(GL_ARRAY_BUFFER, g_vertexBufferId);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, sizeof(VertexType), (char*)NULL + (3 * sizeof(float)));

        // Generate an ID for the index buffer.
        glGenBuffers(1, &g_indexBufferId);

        // Bind the index buffer and load the index data into it.
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_indexBufferId);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_indexCount* sizeof(uint16_t), indices, GL_STATIC_DRAW);

        return true;
}

void ShutdownBuffers()
{
        // Disable the two vertex array attributes.
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);

        // Release the vertex buffer.
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &g_vertexBufferId);

        // Release the index buffer.
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glDeleteBuffers(1, &g_indexBufferId);

        // Release the vertex array object.
        glBindVertexArray(0);
        glDeleteVertexArrays(1, &g_vertexArrayId);

        return;
}

void RenderBuffers()
{
        // Bind the vertex array object that stored all the information about the vertex and index buffers.
        glBindVertexArray(g_vertexArrayId);

        // Render the vertex buffer using the index buffer.
        glDrawElements(GL_TRIANGLES, g_indexCount, GL_UNSIGNED_SHORT, 0);

        return;
}

void CalculateCameraPosition()
{
    VectorType up, position, lookAt;
    float yaw, pitch, roll;
    float rotationMatrix[9];


    // Setup the vector that points upwards.
    up.x = 0.0f;
    up.y = 1.0f;
    up.z = 0.0f;

    // Setup the position of the camera in the world.
    position.x = g_positionX;
    position.y = g_positionY;
    position.z = g_positionZ;

    // Setup where the camera is looking by default.
    lookAt.x = 0.0f;
    lookAt.y = 0.0f;
    lookAt.z = 1.0f;

    // Set the yaw (Y axis), pitch (X axis), and roll (Z axis) rotations in radians.
    pitch = g_rotationX * 0.0174532925f;
    yaw   = g_rotationY * 0.0174532925f;
    roll  = g_rotationZ * 0.0174532925f;

    // Create the rotation matrix from the yaw, pitch, and roll values.
    MatrixRotationYawPitchRoll(rotationMatrix, yaw, pitch, roll);

    // Transform the lookAt and up vector by the rotation matrix so the view is correctly rotated at the origin.
    TransformCoord(lookAt, rotationMatrix);
    TransformCoord(up, rotationMatrix);

    // Translate the rotated camera position to the location of the viewer.
    lookAt.x = position.x + lookAt.x;
    lookAt.y = position.y + lookAt.y;
    lookAt.z = position.z + lookAt.z;

    // Finally create the view matrix from the three updated vectors.
    BuildViewMatrix(position, lookAt, up, g_viewMatrix);
}

void Draw()
{
    static float rotateAngle = 0.0f;

    // Set the color to clear the screen to.
    glClearColor(0.2f, 0.3f, 0.4f, 1.0f);
    // Clear the screen and depth buffer.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Update world matrix to rotate the model
    rotateAngle += PI / 120;
    float rotationMatrixY[16];
    float rotationMatrixZ[16];
    MatrixRotationY(rotationMatrixY, rotateAngle);
    MatrixRotationZ(rotationMatrixZ, rotateAngle);
    MatrixMultiply(g_worldMatrix, rotationMatrixZ, rotationMatrixY);

    // Generate the view matrix based on the camera's position.
    CalculateCameraPosition();

    // Set the color shader as the current shader program and set the matrices that it will use for rendering.
    glUseProgram(g_shaderProgram);
    SetShaderParameters(g_worldMatrix, g_viewMatrix, g_projectionMatrix);

    // Render the model using the color shader.
    RenderBuffers();

    // Present the back buffer to the screen since rendering is complete.
    SwapBuffers(g_deviceContext);
}

// the WindowProc function prototype
LRESULT CALLBACK WindowProc(HWND hWnd,
                         UINT message,
                         WPARAM wParam,
                         LPARAM lParam);

// the entry point for any Windows program
int WINAPI WinMain(HINSTANCE hInstance,
    HINSTANCE hPrevInstance,
    LPTSTR lpCmdLine,
    int nCmdShow)
{
    // the handle for the window, filled by a function
    HWND hWnd;
    // this struct holds information for the window class
    WNDCLASSEX wc;

    // clear out the window class for use
    ZeroMemory(&wc, sizeof(WNDCLASSEX));

    // fill in the struct with the needed information
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc = DefWindowProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)COLOR_WINDOW;
    wc.lpszClassName = _T("Temporary");

    // register the window class
    RegisterClassEx(&wc);

    // create the temporary window for OpenGL extension setup.
    hWnd = CreateWindowEx(WS_EX_APPWINDOW,
                          _T("Temporary"),    // name of the window class
                          _T("Temporary"),   // title of the window
                          WS_OVERLAPPEDWINDOW,    // window style
                          0,    // x-position of the window
                          0,    // y-position of the window
                          640,    // width of the window
                          480,    // height of the window
                          NULL,    // we have no parent window, NULL
                          NULL,    // we aren't using menus, NULL
                          hInstance,    // application handle
                          NULL);    // used with multiple windows, NULL

                                    // Don't show the window.
    ShowWindow(hWnd, SW_HIDE);

    InitializeExtensions(hWnd);

    DestroyWindow(hWnd);
    hWnd = NULL;

    // clear out the window class for use
    ZeroMemory(&wc, sizeof(WNDCLASSEX));

    // fill in the struct with the needed information
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)COLOR_WINDOW;
    wc.lpszClassName = _T("Hello, Engine!");

    // register the window class
    RegisterClassEx(&wc);

    // create the window and use the result as the handle
    hWnd = CreateWindowEx(WS_EX_APPWINDOW,
        _T("Hello, Engine!"),    // name of the window class
        _T("Hello, Engine!"),   // title of the window
        WS_OVERLAPPEDWINDOW,    // window style
        300,    // x-position of the window
        300,    // y-position of the window
        960,    // width of the window
        540,    // height of the window
        NULL,    // we have no parent window, NULL
        NULL,    // we aren't using menus, NULL
        hInstance,    // application handle
        NULL);    // used with multiple windows, NULL

    InitializeOpenGL(hWnd, 960, 540, SCREEN_DEPTH, SCREEN_NEAR, true);

    // display the window on the screen
    ShowWindow(hWnd, nCmdShow);
    SetForegroundWindow(hWnd);

    InitializeShader(hWnd, VS_SHADER_SOURCE_FILE, PS_SHADER_SOURCE_FILE);
    InitializeBuffers();

    // enter the main loop:

    // this struct holds Windows event messages
    MSG msg;

    // wait for the next message in the queue, store the result in 'msg'
    while(GetMessage(&msg, NULL, 0, 0))
    {
        // translate keystroke messages into the right format
        TranslateMessage(&msg);

        // send the message to the WindowProc function
        DispatchMessage(&msg);
    }

    ShutdownBuffers();
    ShutdownShader();
    FinalizeOpenGL(hWnd);

    // return this part of the WM_QUIT message to Windows
    return msg.wParam;
}

// this is the main message handler for the program
LRESULT CALLBACK WindowProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    // sort through and find what code to run for the message given
    switch(message)
    {
    case WM_CREATE:
        {
        }
    case WM_PAINT:
        {
          Draw();
          return 0;
        }
        // this message is read when the window is closed
    case WM_DESTROY:
        {
                // close the application entirely
           PostQuitMessage(0);
           return 0;
        }
    }

    // Handle any messages the switch statement didn't
    return DefWindowProc (hWnd, message, wParam, lParam);
}

