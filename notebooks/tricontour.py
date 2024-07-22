'''import matplotlib.tri as tri

levels = 128

# Transform zenith into degree
zeniths_deg = np.rad2deg(zeniths_t)

# Create triangulation
x = zeniths_deg * np.cos(azimuths_t)
y = zeniths_deg * np.sin(azimuths_t)
triangles = tri.Triangulation(x, y)

# Make plot
fig = plt.figure(figsize=(6,6))
rect = [0, 0, 1, 1]

# Main plot in Cartesian coordinates
ax_cartesian = fig.add_axes(rect, frameon=False, aspect='equal')
ax_cartesian.axis('off')
ctr = ax_cartesian.tricontourf(triangles, brdf_data_t, levels=levels, cmap='turbo')

# Show the contours
#ax_cartesian.tricontour(triangles, brdf_data_t, levels=levels, colors='k', linewidths=0.5)

# Match limits with the full zenith range
ax_cartesian.set_xlim([-90, 90])
ax_cartesian.set_ylim([-90, 90])

# Polar axes
ax_polar = fig.add_axes(rect, polar=True, facecolor="none")
ax_polar.set_rlim([0, 90])          # Cover the full zenith value range
ax_polar.grid(False)                # Hide the polar grid
ax_polar.set_yticklabels([])        # No radial tick labels

# Add the color bar (important: both axes must be adjusted)
fig.colorbar(ctr, ax=[ax_cartesian, ax_polar])

plt.show()
plt.close()'''