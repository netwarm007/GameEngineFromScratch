#include <cmath>
#include <iostream>
#include <string>

#include "AssetLoader.hpp"
#include "OGEX.hpp"

using namespace My;
using namespace std;

template <typename Key, typename T>
static ostream& operator<<(ostream& out,
                           unordered_map<Key, shared_ptr<T>> map) {
    for (const auto& p : map) {
        if (auto ptr = p.second) out << *ptr << endl;
    }

    return out;
}

int main(int, char**) {
    AssetLoader assetLoader;
    assetLoader.Initialize();

    string ogex_text =
        assetLoader.SyncOpenAndReadTextFileToString("Scene/splash.ogex");

    OgexParser ogexParser;
    if (ogexParser.Parse("Node {")) {
        cerr << "FAILED: malformed OpenGEX must report parse failure" << endl;
        return 1;
    }
    const string cameraPrefix =
        "CameraObject $camera { Param (attrib = \"fov\") {float {1.0}} }\n"
        "CameraNode $node { ObjectRef {ref {$camera}}\n";
    const string targetPrefix =
        "Extension (applic = \"MyGameEngine\", type = \"camera_target\") {";
    for (const auto& invalid : {"float {1.0, 2.0}", "string {\"target\"}",
                                "float {1.0, 2.0, 3.0} float {4.0, 5.0, 6.0}",
                                "float {1.0, 2.0, 0x7F800000}"}) {
        if (ogexParser.Parse(cameraPrefix + targetPrefix + invalid + "}}")) {
            cerr << "FAILED: invalid camera target must fail parsing" << endl;
            return 1;
        }
    }
    auto targetScene = ogexParser.Parse(cameraPrefix + targetPrefix +
                                        "float[3] {{0.5, -0.25, 0.83}}}}");
    if (!targetScene || !targetScene->GetFirstCameraNode() ||
        std::abs(targetScene->GetFirstCameraNode()->GetTarget()[0] - 0.5f) >
            0.001f ||
        std::abs(targetScene->GetFirstCameraNode()->GetTarget()[1] + 0.25f) >
            0.001f ||
        std::abs(targetScene->GetFirstCameraNode()->GetTarget()[2] - 0.83f) >
            0.001f) {
        cerr
            << "FAILED: valid world-space camera target must load after failure"
            << endl;
        return 1;
    }
    auto defaultScene = ogexParser.Parse(cameraPrefix + "}");
    if (!defaultScene || !defaultScene->GetFirstCameraNode() ||
        Length(defaultScene->GetFirstCameraNode()->GetTarget()) > 0.001f) {
        cerr << "FAILED: cameras without target extension must retain origin "
                "target"
             << endl;
        return 1;
    }
    {
        shared_ptr<Scene> pScene = ogexParser.Parse(ogex_text);
        if (!pScene) {
            cerr << "FAILED: valid OpenGEX must load after a parse failure"
                 << endl;
            return 1;
        }
    }  // note texture in the scene will be async loaded until process terminate

    assetLoader.Finalize();

    return 0;
}
