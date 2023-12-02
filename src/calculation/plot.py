def generate_very_tight_natural_circular_crowds(num_clusters, cluster_size, cluster_radius):
    all_points = []
    cluster_distance = 5  # Distance between the cluster centers
    for k in range(num_clusters):
        cluster_center = (cluster_distance * k + cluster_radius + 2, cluster_distance * k + cluster_radius + 2)
        for i in range(cluster_size):
            angle_variation = np.random.uniform(-0.2, 0.2)  # slight variation in angle for more natural appearance
            r_variation = cluster_radius * np.random.uniform(-0.7, 0.3)  # increased variation in distance for individuals to be very close together
            angle = 2 * np.pi * i / cluster_size + angle_variation
            r = cluster_radius + r_variation
            x = cluster_center[0] + r * np.cos(angle)
            y = cluster_center[1] + r * np.sin(angle)
            all_points.append((x, y))
    return all_points

# Regenerate very tight natural circular crowd-like clusters
all_points_very_tight_natural_crowd_circular = generate_very_tight_natural_circular_crowds(num_clusters, cluster_size, 1.5)  # Further reduced cluster radius for very tight look

# Plot very tight natural circular crowd-like clusters with angles
plot_clusters_with_angles(all_points_very_tight_natural_crowd_circular, head_angles_circular_engaged_new, "Engaged Clusters - Very Tight Natural Circular Crowd Look")
plot_clusters_with_angles(all_points_very_tight_natural_crowd_circular, head_angles_circular_unengaged_new, "Unengaged Clusters - Very Tight Natural Circular Crowd Look")
