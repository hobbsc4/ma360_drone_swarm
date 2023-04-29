options = {
    "num_drones"             : 70,     # number of drones in the model
    "num_targets"            : 3,      # number of targes in the model
    "domain_height"          : 1000,   # width of the spatial domain, in meters
    "domain_width"           : 1000,   # height of the spatial domain, in meters
    "target_vis_radius"      : 1000,   # target's visibility range, in meters
    "target_weapon_range"    : 400,    # effective weapon range of target, in meters
    "drone_vis_radius"       : 500,    # drone's visibility radius, in meters
    "drone_weapon_radius"    : 100,    # effective weapon radius of drone, in meters
    "drone_accuracy"         : .9,     # probability that drone hits target (between 0 and 1)
    "drone_max_velocity"     : 27,     # meters / second
    "drone_max_acceleration" : 20,     # meters / second^2
    "target_fire_cooldown"   : 2,      # time between target attacks, in seconds
    "target_max_turn_rate"   : 45,     # maximum turning rate of target, in radians/second
    "drone_cost"             : 5000,   # USD per drone
    "target_cost"            : 1e6,    # USD per target
}
