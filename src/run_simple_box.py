import mujoco
from mujoco import viewer

# 加载模型
model = mujoco.MjModel.from_xml_path("models/simple_box.xml")
data = mujoco.MjData(model)

# 打开主动 viewer，自动渲染循环
viewer.launch(model, data)