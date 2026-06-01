import mujoco

# 加载 MuJoCo 模型和运行状态
model = mujoco.MjModel.from_xml_path("models/simple_box.xml")
data = mujoco.MjData(model)

print("=== Model Basic Information ===")
print(f"nq = {model.nq}")
print(f"nv = {model.nv}")
print(f"nu = {model.nu}")

print("\n=== Current State ===")
print(f"qpos = {data.qpos}")
print(f"qvel = {data.qvel}")
print(f"ctrl = {data.ctrl}")

print("\n=== Bodies ===")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = model.body_mass[i]
    inertia = model.body_inertia[i]
    print(f"Body {i}: name={name}, mass={mass},inertia={inertia}")

print("\n=== Joints ===")
for i in range(model.njnt):
    # 获取名称
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    # 获取关节范围
    jrange = model.jnt_range[i]  # shape (2,)
    print(f"Joint {i}: name={name}, range={jrange}")

print("\n=== Actuators ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"Actuator {i}: name={name}")

print("\n=== Geoms ===")
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    size =model.geom_size[i]
    print(f"Geom {i}: name={name}, size={size}")

