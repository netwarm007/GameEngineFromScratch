#include <iostream>

#include "AssetLoader.hpp"
#include "PNG.hpp"
#include "SceneNode.hpp"

using namespace My;
using namespace std;

namespace {
int g_failures = 0;

void Expect(bool condition, const char* what) {
    if (!condition) {
        cerr << "FAILED: " << what << endl;
        g_failures++;
    }
}
}  // namespace

int main(int, char**) {
    auto root = make_shared<BaseSceneNode>("root");
    auto geometry = make_shared<SceneGeometryNode>("geometry");
    auto light = make_shared<SceneLightNode>("light");
    auto camera = make_shared<SceneCameraNode>("camera");
    auto bone = make_shared<SceneBoneNode>("bone");
    auto nested = make_shared<BaseSceneNode>("nested");

    geometry->AddSceneObjectRef("geometry-ref");

    Expect(root->GetChildCount() == 0, "new node has no children");
    Expect(root->GetChildNodes().empty(), "new node has no child nodes");

    auto* nested_raw = nested.get();
    geometry->AppendChild(static_pointer_cast<TreeNode>(nested));

    auto* geometry_raw = geometry.get();
    root->AppendChild(static_pointer_cast<TreeNode>(geometry));
    root->AppendChild(static_pointer_cast<TreeNode>(light));
    root->AppendChild(static_pointer_cast<TreeNode>(camera));
    root->AppendChild(static_pointer_cast<TreeNode>(bone));

    Expect(root->GetChildCount() == 4, "root has four children");
    Expect(root->GetChildren().size() == 4, "TreeNode::GetChildren size");

    const auto children = root->GetChildNodes();
    Expect(children.size() == 4, "GetChildNodes size");
    Expect(children[0].get() == geometry_raw, "children keep insertion order");
    Expect(children[0]->GetParent() == root.get(), "child parent is root");
    Expect(root->GetParent() == nullptr, "root has no parent");
    Expect(children[0]->GetChildNodes().front().get() == nested_raw,
           "nested child is reachable");
    Expect(nested_raw->GetParent() == geometry_raw, "nested parent");

    Expect(root->GetKind() == SceneNodeKind::Empty, "kind of empty node");
    Expect(children[0]->GetKind() == SceneNodeKind::Geometry,
           "kind of geometry node");
    Expect(children[1]->GetKind() == SceneNodeKind::Light,
           "kind of light node");
    Expect(children[2]->GetKind() == SceneNodeKind::Camera,
           "kind of camera node");
    Expect(children[3]->GetKind() == SceneNodeKind::Bone, "kind of bone node");

    Expect(root->GetAnimationClipCount() == 0, "no animation clips");

    const SceneGeometryNode& const_geometry = *geometry;
    Expect(const_geometry.GetSceneObjectRef() == "geometry-ref",
           "const access to object reference");

    // icons used by the Viewer scene graph window
    AssetLoader assetLoader;
    Expect(assetLoader.Initialize() == 0, "AssetLoader initialization");
    PngParser parser;
    for (const char* name : {"empty", "geometry", "light", "camera", "bone"}) {
        const string path = string("Textures/Icons/scene_") + name + ".png";
        Expect(assetLoader.FileExists(path.c_str()), path.c_str());
        Buffer buf = assetLoader.SyncOpenAndReadBinary(path.c_str());
        Image image = parser.Parse(buf);
        Expect(image.Width == 32 && image.Height == 32 && image.data,
               "icon is a decodable 32x32 image");
    }
    assetLoader.Finalize();

    return g_failures;
}
