from typing import List, Callable, Tuple

import pybullet_tools.utils as pu


def get_collision_functor(
    pyb_robot_id,
    moving_bodies: List,
    obstacles: List,
    attachments: List,
    max_distance: float,
    limits_fn: Callable,
    get_obstacle_aabb: Callable,
    joint_indexes: List[int],
    check_aabb_overlap: Callable,
    check_link_pairs: List[Tuple[int, int]],
    **kwargs
):
    def collision_fn(q, verbose=False):

        if limits_fn(q):
            return True

        pu.set_joint_positions(pyb_robot_id, joint_indexes, q)
        for attachment in attachments:
            attachment.assign()
        # wait_for_duration(1e-2)
        get_moving_aabb = pu.cached_fn(
            pu.get_buffered_aabb,
            cache=True,
            max_distance=max_distance / 2.0,
            **kwargs,
        )

        for link1, link2 in check_link_pairs:
            # Self-collisions should not have the max_distance parameter
            if (
                check_aabb_overlap(
                    get_moving_aabb(pyb_robot_id),
                    get_moving_aabb(pyb_robot_id),
                )
            ) and pu.pairwise_link_collision(
                pyb_robot_id, link1, pyb_robot_id, link2
            ):  # , **kwargs):
                return True

        for body1, body2 in pu.product(moving_bodies, obstacles):
            if (
                check_aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))
            ) and pu.pairwise_collision(body1, body2, **kwargs):
                return True
        return False

    return collision_fn


####################################
# the following is the same as the collision_fn,
# except it returns all colliding points.
# i.e. it has NO early return, but can get more info out of it.


def pairwise_link_collision_get_points(
    body1, link1, body2, link2=pu.BASE_LINK, **kwargs
):
    return pu.get_closest_points(body1, body2, link1=link1, link2=link2, **kwargs)


def body_collision_get_points(body1, body2, **kwargs):
    return pu.get_closest_points(body1, body2, **kwargs)


def any_link_pair_collision_get_points(body1, links1, body2, links2=None, **kwargs):
    if links1 is None:
        links1 = pu.get_all_links(body1)
    if links2 is None:
        links2 = pu.get_all_links(body2)
    colliding_points = []
    for link1, link2 in pu.product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        colliding_points.extend(
            pairwise_link_collision_get_points(body1, link1, body2, link2, **kwargs)
        )
    return colliding_points


def pairwise_collision_get_points(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = pu.expand_links(body1)
        body2, links2 = pu.expand_links(body2)
        return any_link_pair_collision_get_points(body1, links1, body2, links2, **kwargs)
    return body_collision_get_points(body1, body2, **kwargs)


def get_colliding_points_functor(
    pyb_robot_id,
    moving_bodies: List,
    obstacles: List,
    attachments: List,
    max_distance: float,
    limits_fn: Callable,
    get_obstacle_aabb: Callable,
    joint_indexes: List[int],
    check_aabb_overlap: Callable,
    check_link_pairs: List[Tuple[int, int]],
    **kwargs
):
    def collision_fn(q, verbose=False):
        assert not limits_fn(q)

        pu.set_joint_positions(pyb_robot_id, joint_indexes, q)
        for attachment in attachments:
            attachment.assign()
        # wait_for_duration(1e-2)
        get_moving_aabb = pu.cached_fn(
            pu.get_buffered_aabb,
            cache=True,
            max_distance=max_distance / 2.0,
            **kwargs,
        )

        colliding_points = []

        for link1, link2 in check_link_pairs:
            # Self-collisions should not have the max_distance parameter
            if check_aabb_overlap(
                get_moving_aabb(pyb_robot_id),
                get_moving_aabb(pyb_robot_id),
            ):
                colliding_points.extend(
                    pairwise_link_collision_get_points(
                        pyb_robot_id, link1, pyb_robot_id, link2
                    )
                )
        for body1, body2 in pu.product(moving_bodies, obstacles):
            if check_aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2)):
                colliding_points.extend(pairwise_collision_get_points(body1, body2, **kwargs))
        return colliding_points

    return collision_fn
