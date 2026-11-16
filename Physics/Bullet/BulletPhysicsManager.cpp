#include "BulletPhysicsManager.hpp"

#include <iostream>

#include "ISceneManager.hpp"

using namespace My;
using namespace std;

int BulletPhysicsManager::Initialize() {
    // Build the broadphase
    m_btBroadphase = new btDbvtBroadphase();

    // Set up the collision configuration and dispatcher
    m_btCollisionConfiguration = new btDefaultCollisionConfiguration();
    m_btDispatcher = new btCollisionDispatcher(m_btCollisionConfiguration);

    // The actual physics solver
    m_btSolver = new btSequentialImpulseConstraintSolver;

    // The world
    m_btDynamicsWorld = new btDiscreteDynamicsWorld(
        m_btDispatcher, m_btBroadphase, m_btSolver, m_btCollisionConfiguration);
    m_btDynamicsWorld->setGravity(btVector3(0.0f, 0.0f, -9.8f));

    return 0;
}

void BulletPhysicsManager::Finalize() {
    // Clean up
    ClearRigidBodies();

    delete m_btDynamicsWorld;
    delete m_btSolver;
    delete m_btDispatcher;
    delete m_btCollisionConfiguration;
    delete m_btBroadphase;
    m_nSceneRevision = 0;
}

void BulletPhysicsManager::Tick() {
    auto rev = m_Scenes.GetSceneRevision();
    if (m_nSceneRevision != rev) {
        if (CreateRigidBodies() == 0) {
            m_nSceneRevision = rev;
        }
    }

    m_btDynamicsWorld->stepSimulation(1.0f / 60.0f, 10);
    PublishTransforms();
}

namespace {
btTransform ToBulletTransform(const Matrix4X4f& trans) {
    btTransform result;
    result.setIdentity();
    result.setOrigin(
        btVector3(trans.data[3][0], trans.data[3][1], trans.data[3][2]));
    result.setBasis(
        btMatrix3x3(trans.data[0][0], trans.data[1][0], trans.data[2][0],
                    trans.data[0][1], trans.data[1][1], trans.data[2][1],
                    trans.data[0][2], trans.data[1][2], trans.data[2][2]));
    return result;
}

btVector3 ReadPosition(const SceneObjectVertexArray& array, size_t index) {
    if (array.GetDataType() == VertexDataType::kVertexDataTypeDouble3) {
        const auto* d = static_cast<const double*>(array.GetData()) + index * 3;
        return {d[0], d[1], d[2]};
    }
    const auto* f = static_cast<const float*>(array.GetData()) + index * 3;
    return {f[0], f[1], f[2]};
}

size_t ReadIndex(const SceneObjectIndexArray& array, size_t index) {
    switch (array.GetIndexType()) {
        case IndexDataType::kIndexDataTypeInt8:
            return static_cast<const uint8_t*>(array.GetData())[index];
        case IndexDataType::kIndexDataTypeInt16:
            return static_cast<const uint16_t*>(array.GetData())[index];
        case IndexDataType::kIndexDataTypeInt32:
            return static_cast<const uint32_t*>(array.GetData())[index];
        default:
            return static_cast<size_t>(
                static_cast<const uint64_t*>(array.GetData())[index]);
    }
}
}  // namespace

std::unique_ptr<btTriangleMesh> BulletPhysicsManager::BuildTriangleMesh(
    const SceneObjectGeometry& geometry) const {
    auto mesh = geometry.GetMesh().lock();
    if (!mesh || mesh->GetPrimitiveType() != PrimitiveType::kPrimitiveTypeTriList) {
        return nullptr;
    }

    const SceneObjectVertexArray* positions = nullptr;
    for (uint32_t i = 0; i < mesh->GetVertexPropertiesCount(); i++) {
        const auto& array = mesh->GetVertexPropertyArray(i);
        if (array.GetAttributeName() == "position") {
            positions = &array;
            break;
        }
    }
    const bool supported =
        positions &&
        (positions->GetDataType() == VertexDataType::kVertexDataTypeFloat3 ||
         positions->GetDataType() == VertexDataType::kVertexDataTypeDouble3);
    if (!supported) return nullptr;

    auto triangles = std::make_unique<btTriangleMesh>(true, false);
    const size_t vertexCount = positions->GetVertexCount();
    for (size_t group = 0; group < mesh->GetIndexGroupCount(); group++) {
        const auto& indices = mesh->GetIndexArray(group);
        for (size_t i = 0; i + 2 < indices.GetIndexCount(); i += 3) {
            const size_t a = ReadIndex(indices, i);
            const size_t b = ReadIndex(indices, i + 1);
            const size_t c = ReadIndex(indices, i + 2);
            if (a >= vertexCount || b >= vertexCount || c >= vertexCount) {
                return nullptr;
            }
            triangles->addTriangle(ReadPosition(*positions, a),
                                   ReadPosition(*positions, b),
                                   ReadPosition(*positions, c));
        }
    }
    return triangles->getNumTriangles() > 0 ? std::move(triangles) : nullptr;
}

void BulletPhysicsManager::CreateRigidBody(
    const std::shared_ptr<SceneGeometryNode>& sceneNode,
    const SceneObjectGeometry& geometry) {
    assert(sceneNode);
    auto& node = *sceneNode;
    DeleteRigidBody(node);
    if (!node.PhysicsEnabled()) return;

    const float* param = geometry.CollisionParameters();
    const auto& physics = geometry.PhysicsProperties();
    const auto trans = node.GetCalculatedTransform();
    btTransform startTransform = ToBulletTransform(*trans);

    btCollisionShape* shape = nullptr;
    bool dynamicByDefault = false;
    switch (geometry.CollisionType()) {
        case SceneObjectCollisionType::kSceneObjectCollisionTypeSphere:
            shape = new btSphereShape(param[0]);
            dynamicByDefault = true;
            break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypeBox:
            shape = new btBoxShape(btVector3(param[0], param[1], param[2]));
            break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypePlane:
            shape = new btStaticPlaneShape(
                btVector3(param[0], param[1], param[2]), param[3]);
            break;
        case SceneObjectCollisionType::kSceneObjectCollisionTypeBvhMesh: {
            // The mesh is baked in object space; node scale is applied as
            // shape scaling so the body transform stays orthonormal.
            auto triangles = BuildTriangleMesh(geometry);
            if (!triangles) {
                cerr << "[BulletPhysicsManager] " << node.GetName()
                     << ": bvhmesh collision needs a triangle list mesh"
                     << endl;
                return;
            }
            btMatrix3x3 basis = startTransform.getBasis();
            btVector3 scaling(1, 1, 1);
            for (int column = 0; column < 3; column++) {
                btVector3 axis(basis[0][column], basis[1][column],
                               basis[2][column]);
                const btScalar length = axis.length();
                if (length > SIMD_EPSILON) {
                    scaling[column] = length;
                    for (int row = 0; row < 3; row++) {
                        basis[row][column] /= length;
                    }
                }
            }
            startTransform.setBasis(basis);
            auto* bvh = new btBvhTriangleMeshShape(triangles.get(), true);
            bvh->setLocalScaling(scaling);
            m_btTriangleMeshes.push_back(std::move(triangles));
            shape = bvh;
        } break;
        default:
            return;
    }
    m_btCollisionShapes.push_back(shape);

    const bool dynamic = !shape->isConcave() &&
                         (physics.bodyType == SceneObjectBodyType::kDynamic ||
                          (physics.bodyType == SceneObjectBodyType::kAuto &&
                           dynamicByDefault));
    const btScalar mass = dynamic ? physics.mass : 0.0f;
    btVector3 inertia(0.0f, 0.0f, 0.0f);
    if (dynamic) shape->calculateLocalInertia(mass, inertia);

    auto* motionState = new btDefaultMotionState(startTransform);
    btRigidBody::btRigidBodyConstructionInfo rigidBodyCI(mass, motionState,
                                                         shape, inertia);
    rigidBodyCI.m_friction = physics.friction;
    rigidBodyCI.m_restitution = physics.restitution;
    auto* rigidBody = new btRigidBody(rigidBodyCI);

    if (dynamic) {
        btVector3 velocity(physics.linearVelocity[0], physics.linearVelocity[1],
                           physics.linearVelocity[2]);
        if (physics.planeNormalAxis >= 0) {
            // Bullet's factors only scale impulses, so the initial velocity
            // must not leave the plane either.
            btVector3 linear(1, 1, 1);
            btVector3 angular(0, 0, 0);
            linear[physics.planeNormalAxis] = 0;
            angular[physics.planeNormalAxis] = 1;
            velocity[physics.planeNormalAxis] = 0;
            rigidBody->setLinearFactor(linear);
            rigidBody->setAngularFactor(angular);
        }
        rigidBody->setLinearVelocity(velocity);
    }
    m_btDynamicsWorld->addRigidBody(rigidBody);

    m_RigidBodies.emplace(&node, BodyBinding{sceneNode, rigidBody});
    node.SetSimulationTransform(GetRigidBodyTransform(*rigidBody));
}

bool BulletPhysicsManager::UpdateRigidBodyTransform(SceneGeometryNode& node) {
    auto it = m_RigidBodies.find(&node);
    if (it == m_RigidBodies.end()) {
        return false;
    }
    const auto trans = node.GetCalculatedTransform();
    auto* rigidBody = it->second.body;
    const btTransform _trans = ToBulletTransform(*trans);
    rigidBody->setWorldTransform(_trans);
    rigidBody->getMotionState()->setWorldTransform(_trans);
    m_btDynamicsWorld->updateSingleAabb(rigidBody);
    node.SetSimulationTransform(GetRigidBodyTransform(*rigidBody));
    return true;
}

void BulletPhysicsManager::DeleteRigidBody(SceneGeometryNode& node) {
    auto it = m_RigidBodies.find(&node);
    if (it != m_RigidBodies.end()) {
        auto* rigidBody = it->second.body;
        m_btDynamicsWorld->removeRigidBody(rigidBody);

        delete rigidBody->getMotionState();
        delete rigidBody;
        node.ClearSimulationTransform();
        m_RigidBodies.erase(it);
    } else {
        node.ClearSimulationTransform();
    }
}

int BulletPhysicsManager::CreateRigidBodies() {
    ClearRigidBodies();
    m_pSimulationScene = m_Scenes.GetSceneForPhysicalSimulation();
    const auto& scene = m_pSimulationScene;
    if (!scene) {
        cerr << "[BulletPhysicsManager] Scene revision has no scene" << endl;
        return -1;
    }

    // Geometries
    for (const auto& _it : scene->GeometryNodes) {
        auto pGeometryNode = _it.second.lock();
        if (pGeometryNode) {
            auto pGeometry =
                scene->GetGeometry(pGeometryNode->GetSceneObjectRef());
            assert(pGeometry);

            CreateRigidBody(pGeometryNode, *pGeometry);
        }
    }

    return 0;
}

void BulletPhysicsManager::ClearRigidBodies() {
    while (!m_RigidBodies.empty()) {
        DeleteRigidBody(*m_RigidBodies.begin()->first);
    }

    for (auto shape : m_btCollisionShapes) {
        delete shape;
    }

    m_btCollisionShapes.clear();
    m_btTriangleMeshes.clear();
    m_pSimulationScene.reset();
}

Matrix4X4f BulletPhysicsManager::GetRigidBodyTransform(
    const btRigidBody& rigidBody) const {
    Matrix4X4f result;
    btTransform trans;
    rigidBody.getMotionState()->getWorldTransform(trans);
    auto basis = trans.getBasis();
    auto origin = trans.getOrigin();
    BuildIdentityMatrix(result);
    result.data[0][0] = static_cast<float>(basis[0][0]);
    result.data[1][0] = static_cast<float>(basis[0][1]);
    result.data[2][0] = static_cast<float>(basis[0][2]);
    result.data[0][1] = static_cast<float>(basis[1][0]);
    result.data[1][1] = static_cast<float>(basis[1][1]);
    result.data[2][1] = static_cast<float>(basis[1][2]);
    result.data[0][2] = static_cast<float>(basis[2][0]);
    result.data[1][2] = static_cast<float>(basis[2][1]);
    result.data[2][2] = static_cast<float>(basis[2][2]);
    result.data[3][0] = static_cast<float>(origin.getX());
    result.data[3][1] = static_cast<float>(origin.getY());
    result.data[3][2] = static_cast<float>(origin.getZ());

    return result;
}

void BulletPhysicsManager::PublishTransforms() {
    for (const auto& [node, binding] : m_RigidBodies) {
        node->SetSimulationTransform(GetRigidBodyTransform(*binding.body));
    }
}

bool BulletPhysicsManager::ApplyCentralForce(SceneGeometryNode& node,
                                            Vector3f force) {
    auto it = m_RigidBodies.find(&node);
    if (it == m_RigidBodies.end()) {
        return false;
    }
    auto* _rigidBody = it->second.body;
    btVector3 _force(force[0], force[1], force[2]);
    _rigidBody->activate(true);
    _rigidBody->applyCentralForce(_force);
    return true;
}