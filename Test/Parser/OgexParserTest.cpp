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
    const string geometryPrefix =
        "GeometryObject $geo { Mesh (primitive = \"triangles\") {"
        "VertexArray (attrib = \"position\") {float[3] {{0,0,0},{1,0,0},"
        "{0,1,0}}}}\n";
    const string geometrySuffix = "}\nGeometryNode $node { Name {string {\"node\"}} "
                                  "ObjectRef {ref {$geo}} }\n";
    const string ext = "Extension (applic = \"MyGameEngine\", type = ";
    for (const auto& invalid :
         {"\"physics_material\") {float {-1.0, 0.5}}",
          "\"physics_material\") {float {0.5}}",
          "\"rigid_body\") {string {\"kinematic\"} float {1, 0, 0, 0}}",
          "\"rigid_body\") {string {\"dynamic\"} float {0, 0, 0, 0}}",
          "\"motion_constraint\") {string {\"plane\"} float {0.5, 0.5, 0.5}}",
          "\"collision\") {string {\"sphere\"}}"}) {
        if (ogexParser.Parse(geometryPrefix + ext + invalid + geometrySuffix)) {
            cerr << "FAILED: invalid physics extension must fail parsing"
                 << endl;
            return 1;
        }
    }
    auto physicsScene = ogexParser.Parse(
        geometryPrefix + ext + "\"collision\") {string {\"bvhmesh\"}}\n" + ext +
        "\"physics_material\") {float {0.4, 0.95}}\n" + ext +
        "\"rigid_body\") {string {\"dynamic\"} float {2, 1.5, 0, 0}}\n" + ext +
        "\"motion_constraint\") {string {\"plane\"} float {0, 1, 0}}" +
        geometrySuffix);
    auto physicsGeometry =
        physicsScene ? physicsScene->GetGeometry("geo") : nullptr;
    if (!physicsGeometry) {
        cerr << "FAILED: valid physics extensions must load" << endl;
        return 1;
    }
    const auto& physics = physicsGeometry->PhysicsProperties();
    if (physicsGeometry->CollisionType() !=
            SceneObjectCollisionType::kSceneObjectCollisionTypeBvhMesh ||
        std::abs(physics.friction - 0.4f) > 1e-5f ||
        std::abs(physics.restitution - 0.95f) > 1e-5f ||
        physics.bodyType != SceneObjectBodyType::kDynamic ||
        std::abs(physics.mass - 2.0f) > 1e-5f ||
        std::abs(physics.linearVelocity[0] - 1.5f) > 1e-5f ||
        physics.planeNormalAxis != 1) {
        cerr << "FAILED: physics extension values were not preserved" << endl;
        return 1;
    }
    auto deepNodes = [](int depth) {
        string text;
        for (int i = 0; i < depth; i++) text += "Node {";
        for (int i = 0; i < depth; i++) text += "}";
        return text;
    };
    if (!ogexParser.Parse(deepNodes(64)) || ogexParser.Parse(deepNodes(5000))) {
        cerr << "FAILED: nesting must be bounded without rejecting normal "
                "scenes"
             << endl;
        return 1;
    }
    if (ogexParser.Parse(string("Node {}\0Node {}", 15))) {
        cerr << "FAILED: embedded NUL must fail parsing" << endl;
        return 1;
    }
    const string indexPrefix =
        "GeometryObject $geo { Mesh (primitive = \"triangles\") {"
        "VertexArray (attrib = \"position\") {float[3] {{0,0,0},{1,0,0},"
        "{0,1,0}}} IndexArray {";
    for (const auto& invalid : {"unsigned_int32[3] {{0, 1, 3}}",
                                "unsigned_int8[3] {{0, 1, 255}}",
                                "unsigned_int8[3] {{0, 1, 257}}"}) {
        if (ogexParser.Parse(indexPrefix + invalid + "}}}")) {
            cerr << "FAILED: invalid index array must fail parsing" << endl;
            return 1;
        }
    }
    if (!ogexParser.Parse(indexPrefix + "unsigned_int16[3] {{0, 1, 2}}}}}")) {
        cerr << "FAILED: valid index array must load" << endl;
        return 1;
    }
    if (ogexParser.Parse(geometryPrefix + ext +
                         "\"collision\") {string {\"cone\"}}" +
                         geometrySuffix)) {
        cerr << "FAILED: unknown collision shape must fail parsing" << endl;
        return 1;
    }
    const string animationPrefix =
        "Node $n { Translation %t (kind = \"x\") {float {0}}\n"
        "Animation { Track (target = %t) { Time (curve = \"linear\") {Key "
        "{float {0, 1, 2}}} Value (curve = \"linear\") {Key {float {";
    if (ogexParser.Parse(animationPrefix + "0, 1}}}}}}") ||
        ogexParser.Parse(animationPrefix + "0, 1, 0x7F800000}}}}}}")) {
        cerr << "FAILED: inconsistent animation keys must fail parsing" << endl;
        return 1;
    }
    if (!ogexParser.Parse(animationPrefix + "0, 1, 4}}}}}}")) {
        cerr << "FAILED: valid animation must load" << endl;
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
