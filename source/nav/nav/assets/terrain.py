"""保留实际生成的静态地形网格，供导航的几何碰撞查询使用。"""

from isaaclab.terrains import TerrainImporter
from isaaclab.utils.warp import convert_to_warp_mesh


class CollisionTerrainImporter(TerrainImporter):
    """用于当前高度场地形，保留与 USD 地形一致的 Warp 查询网格。

    当前地形在世界原点生成，父 prim 不施加额外变换，且运行期间不移动。
    高度场下方视为实体；不适用于桥梁、隧道或悬空网格。
    """

    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        if name == "terrain":
            # 复用已经生成的网格，不再次随机生成地形。
            self.collision_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
            self.collision_mesh_top = float(mesh.bounds[1, 2]) + 1.0

