from vehicles.entity.base_object import BaseObject
from vehicles.entity.angles import Angle, AngularType
from vehicles.entity.senses import SensorType, SensorShape, Sense


class Vehicle(BaseObject):
    def __init__(self,
                 mass: int,
                 position: tuple[int, int],
                 size: tuple[int, int],
                 facing_point: tuple[int, int],
                 sense:Sense):
        super().__init__(mass=mass, position=position, size=size, facing_point=facing_point)

        self.sense = sense

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    positions = []
    pointers = []
    bounding_boxes = []

    sense_obj = Sense(type=SensorType.SIGHT,
                      shape=SensorShape.CONE,
                      range=10,
                      field_of_view=Angle(type=AngularType.RADIANS, value=0.5),
                      noise=0.12,
                      xray=False
                      )

    ent = Vehicle(
        mass=2,
        position=(-10.0, -10.0),
        size=(5.0, 5.0),
        facing_point=(4.0, 3.0),
        sense=sense_obj
    )

    print(ent)
    print(f"Initial heading: {ent.heading.value:.4f} rad ({ent.direction.value:.2f}°)")

    plt.figure(figsize=(20, 20))
    positions.append(ent.position.copy())
    pointers.append(ent.facing_point.copy())
    bounding_boxes.append(ent.bounding_box.copy())

    # Turn & move animation loop
    for i in range(18):
        if i < 2:
            # Clockwise turns
            turn_angle = 0.2 if i < 5 else 0.25
        else:
            # Counter-clockwise turns
            turn_angle = -0.2 if i < 10 else -0.25

        _turn = Angle(AngularType.RADIANS, turn_angle)

        ent.turn(_turn)
        ent.move()

        positions.append(ent.position.copy())
        pointers.append(ent.facing_point.copy())
        bounding_boxes.append(ent.bounding_box.copy())

        print(f"Step {i}: pos={ent.position}, facing_point={ent.facing_point}, "
              f"heading={ent.heading.value:.4f} rad ({ent.direction.value:.2f}°)")

    # Visualization loop
    for i in range(len(positions)):
        ax = plt.subplot(5, 4, i + 1)

        pos = positions[i]
        h = pointers[i]
        bb = bounding_boxes[i]

        # Draw position dot
        plt.plot(pos[0], pos[1], marker="o", color="blue", markersize=20)

        # Draw facing_point dot
        plt.plot(h[0], h[1], marker=".", color="orange", markersize=10)

        # Draw connection line
        plt.plot([pos[0], h[0]], [pos[1], h[1]], alpha=0.5, linewidth=2)

        # Draw bounding box corners
        if bb is not None:
            # BB should be (4, 2) array of corners
            plt.plot(bb[:, 0], bb[:, 1], color="red", linewidth=2)
            for j, (x, y) in enumerate(bb):
                plt.plot(x, y, marker="s", color="green", markersize=30, alpha=0.1)


        ax.set_title(f"Step {i + 1}: heading={ent.heading.value:.4f} rad")
        ax.set_aspect("equal")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)

    plt.tight_layout()
    plt.show()

    plt.plot(positions, color="blue", alpha=0.66, markersize=20)
    plt.plot(pointers, color="orange", alpha=0.5, markersize=10)
    plt.show()
    print("\n=== Final State ===")
    print(f"Position: {ent.position}")
    print(f"Facing Point: {ent.facing_point}")
    print(f"Heading: {ent.heading.value:.4f} rad ({ent.direction.value:.2f}°)")
    print(f"Bounding Box Shape: {ent.bounding_box.shape}")