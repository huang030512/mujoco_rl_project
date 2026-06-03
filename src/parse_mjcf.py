import xml.etree.ElementTree as ET

# 加载 MJCF XML
tree = ET.parse("models/mjcf_structure_demo.xml")
root = tree.getroot()

# 遍历 worldbody
worldbody = root.find("worldbody")

def print_body(body, indent=0):
    prefix = " " * indent
    name = body.attrib.get("name", "Unnamed")
    pos = body.attrib.get("pos", "0 0 0")
    print(f"{prefix}Body: {name}, pos={pos}")
    
    for geom in body.findall("geom"):
        gname = geom.attrib.get("name", "UnnamedGeom")
        gtype = geom.attrib.get("type", "box")
        gsize = geom.attrib.get("size", "")
        print(f"{prefix}  Geom: {gname}, type={gtype}, size={gsize}")
    
    for freejoint in body.findall("freejoint"):
        jname = freejoint.attrib.get("name", "UnnamedFreeJoint")
        print(f"{prefix}  Joint: {jname}, type=freejoint")

    for joint in body.findall("joint"):
        jname = joint.attrib.get("name", "UnnamedJoint")
        jtype = joint.attrib.get("type", "hinge")
        print(f"{prefix}  Joint: {jname}, type={jtype}")
    
    # 递归打印子 body
    for child in body.findall("body"):
        print_body(child, indent + 4)

# 打印 worldbody 下所有 body
for body in worldbody.findall("body"):
    print_body(body)
