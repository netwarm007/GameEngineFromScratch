#include "ViewerLogic.hpp"

#include "IApplication.hpp"
#include "SceneManager.hpp"
#include "geommath.hpp"

using namespace My;
using namespace std;

int ViewerLogic::Initialize() {
    int result;

    cout << "[ViewerLogic] Viewer Logic Initialize" << endl;

    if (g_pApp->GetCommandLineArgumentsCount() > 1) {
        auto scene_filename = g_pApp->GetCommandLineArgument(1);
        cout << "[ViewerLogic] Loading Scene: " << scene_filename << endl;
        result = g_pSceneManager->LoadScene(scene_filename);
    } else {
        cout << "[ViewerLogic] Loading Splash Scene" << endl;
        result = g_pSceneManager->LoadScene("Scene/splash.ogex");
    }

    return result;
}

void ViewerLogic::Finalize() { cout << "[ViewerLogic] Finalize" << endl; }

void ViewerLogic::Tick() {}

#ifdef DEBUG
void ViewerLogic::DrawDebugInfo() {}
#endif

void ViewerLogic::OnLeftKeyDown() {
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene->GetFirstCameraNode();
    if (pCameraNode) {
        auto local_axis = pCameraNode->GetLocalAxis();
        Vector3f camera_x_axis;
        memcpy(camera_x_axis.data, local_axis[0], sizeof(camera_x_axis));

        // move camera along its local axis x direction
        pCameraNode->MoveBy(camera_x_axis);
    }
}

void ViewerLogic::OnRightKeyDown() {
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene->GetFirstCameraNode();
    if (pCameraNode) {
        auto local_axis = pCameraNode->GetLocalAxis();
        Vector3f camera_x_axis;
        memcpy(camera_x_axis.data, local_axis[0], sizeof(camera_x_axis));

        // move along camera local axis -x direction
        pCameraNode->MoveBy(camera_x_axis * -1.0f);
    }
}

void ViewerLogic::OnUpKeyDown() {
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene->GetFirstCameraNode();
    if (pCameraNode) {
        auto local_axis = pCameraNode->GetLocalAxis();
        Vector3f camera_y_axis;
        memcpy(camera_y_axis.data, local_axis[1], sizeof(camera_y_axis));

        // move camera along its local axis y direction
        pCameraNode->MoveBy(camera_y_axis);
    }
}

void ViewerLogic::OnDownKeyDown() {
    auto& scene = g_pSceneManager->GetSceneForRendering();
    auto pCameraNode = scene->GetFirstCameraNode();
    if (pCameraNode) {
        auto local_axis = pCameraNode->GetLocalAxis();
        Vector3f camera_y_axis;
        memcpy(camera_y_axis.data, local_axis[1], sizeof(camera_y_axis));

        // move camera along its local axis -y direction
        pCameraNode->MoveBy(camera_y_axis * -1.0f);
    }
}

void ViewerLogic::OnAnalogStick(int id, float deltaX, float deltaY) {
    if (id == 1) {
        auto& scene = g_pSceneManager->GetSceneForRendering();
        auto pCameraNode = scene->GetFirstCameraNode();
        if (pCameraNode) {
            auto screen_width = g_pApp->GetConfiguration().screenWidth;
            auto screen_height = g_pApp->GetConfiguration().screenHeight;
            // move camera along its local axis -y direction
            pCameraNode->RotateBy(deltaX / screen_width * PI,
                                  deltaY / screen_height * PI, 0.0f);
        }
    }
}
