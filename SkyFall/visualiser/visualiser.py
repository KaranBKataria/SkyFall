import os
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

from SkyFall.utils.global_variables import *

class Visualiser:
    def __init__(self, times, trajectory_cartesian, trajectory_LLA, crash_lon_list):
        """
        Initialize the Visualiser class for satellite orbit decay visualization.
        
        Parameters:
        - times: Array-like, time points (seconds)
        - xs: Array-like, satellite position along equatorial direction (meters)
        - ys: Array-like, satellite height above Earth's surface (meters)
        - vxs: Array-like, velocity along equatorial direction (m/s, optional)
        - vys: Array-like, velocity perpendicular to surface (m/s, optional)
        - crash_x_list: List of arrays, predicted crash positions (meters, optional)
        """
        
        self.R_earth = R_e  # Earth radius in meters

        self.times = times
        self.xs = trajectory_cartesian[:,0]
        self.ys = trajectory_cartesian[:,1]
        self.vxs = trajectory_cartesian[:,2]
        self.vys = trajectory_cartesian[:,3]

        # NB: Longitude and Latitude is in degrees (NOT RADIANS)
        self.lat = trajectory_LLA[:,0]
        self.lon = trajectory_LLA[:,1]
        self.altitude = trajectory_LLA[:,-1]

        # Adjust crash_lon_list length to match times
        crash_lon_list = self._adjust_crash_lon_list(crash_lon_list, len(times))
        n_steps = crash_lon_list.shape[0]
        self.crash_lon_list = [crash_lon_list[i, :, 0] for i in range(n_steps)]


        # Debug: Check data lengths and values
        if len(self.times) != len(self.xs) or len(self.xs) != len(self.ys):
            raise ValueError(f"Length mismatch: times ({len(self.times)}), xs ({len(self.xs)}), ys ({len(self.ys)})")
        
        # if self.crash_x_list is not None and len(self.crash_x_list) != len(self.times):
        #     raise ValueError(f"Length mismatch: crash_x_list ({len(self.crash_x_list)}), times ({len(self.times)})")

    def _adjust_crash_lon_list(self, crash_lon_list, target_length):
        """
        Adjust crash_lon_list length to match target_length by repeating elements.
        
        Args:
            crash_lon_list: Array of crash longitude predictions (shape: [sets, samples, dims])
            target_length: Desired length (equal to len(times))
        
        Returns:
            Array with length equal to target_length along the first axis
        """
        if len(crash_lon_list) == 0:
            raise ValueError("crash_lon_list cannot be empty")
            
        repeat_times = 5  # Number of times to repeat each prediction set
        base_length = len(crash_lon_list) * repeat_times
        
        # Step 1: Repeat each prediction set repeat_times times along axis 0
        adjusted_list = np.repeat(crash_lon_list, repeat_times, axis=0)
        
        # Step 2: Add extra elements if needed, copying the last prediction set
        extra_length = target_length - base_length
        if extra_length > 0:
            # Use the last prediction set of the original crash_lon_list
            extra_elements = np.repeat(crash_lon_list[-1][np.newaxis, :], extra_length, axis=0)
            adjusted_list = np.concatenate([adjusted_list, extra_elements], axis=0)

        return adjusted_list

    def _convert_to_lonlat(self):
        """
        Convert xs (equatorial distance) and ys (height) to longitude and latitude.
        Assumes motion along equator (latitude = 0).
        """
        # Convert xs to longitude (degrees)
        lons = self.xs / self.R_earth * 180 / np.pi
        # Assume motion along equator, latitude = 0
        lats = np.zeros_like(lons)
        # Heights (in meters)
        heights = self.ys
        # Check for NaN or invalid values
        if np.any(np.isnan(heights)) or np.any(np.isinf(heights)):
            print("Warning: NaN or Inf values found in heights. Check input data.")
        if np.any(np.isnan(self.times)) or np.any(np.isinf(self.times)):
            print("Warning: NaN or Inf values found in times. Check input data.")
        # Filter points where height > 0 (valid orbit)
        valid = heights > 0
        if not np.any(valid):
            print("Warning: No valid heights (ys > 0) found. Check input data.")
        # Convert crash_x_list to longitudes if provided
        crash_lons_list = None
        if self.crash_x_list is not None:
            crash_lons_list = [np.array(crash_xs) / self.R_earth * 180 / np.pi for crash_xs in self.crash_x_list]
            for i, crash_lons in enumerate(crash_lons_list):
                if np.any(np.isnan(crash_lons)) or np.any(np.isinf(crash_lons)):
                    print(f"Warning: NaN or Inf values found in crash_x_list[{i}].")
        return lons[valid], lats[valid], heights[valid], self.times[valid], crash_lons_list

    def plot_height_vs_time(self, figsize=(8, 6), title='Predictor altitude estimate against time', title_fontsize=14,
                            label_fontsize=12, tick_fontsize=10, line_color='blue', line_width=2,
                            show_grid=True, show_legend=True):
        """
        Plot height vs time as a standalone figure.
        
        Parameters:
        - figsize: tuple, figure size (width, height) in inches (default: (8, 6))
        - title: str, plot title (default: 'Height vs Time')
        - title_fontsize: int, title font size (default: 14)
        - label_fontsize: int, axis label font size (default: 12)
        - tick_fontsize: int, tick label font size (default: 10)
        - line_color: str, color of the height line (default: 'blue')
        - line_width: float, width of the height line (default: 2)
        - show_grid: bool, whether to show grid (default: True)
        - show_legend: bool, whether to show legend (default: True)
        """
        # Convert coordinates
        # lons, lats, heights, times, _ = self._convert_to_lonlat()
        heights = self.altitude
        times = self.times

        # Create figure
        fig, ax_height = plt.subplots(figsize=figsize)

        # Height vs Time
        ax_height.plot(times, heights/1e3, marker='.', color=line_color, linewidth=line_width, label='Altitude (km)')
        ax_height.set_xlabel('Time (seconds)', fontsize=label_fontsize)
        ax_height.set_ylabel('Altitude (km)', fontsize=label_fontsize)
        ax_height.set_title(title, fontsize=title_fontsize)
        ax_height.axhline(min(heights/1e3), label='Predictor termination', linestyle='--', color='red')

        if show_grid:
            ax_height.grid(True)
        if show_legend:
            ax_height.legend()
        if len(heights) > 0 and len(times) > 0:
            ax_height.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
                              max(times) + 0.1 * (max(times) - min(times)))
            # ax_height.set_ylim(min(heights/1e3) * 0.9, max(heights/1e3) * 1.1)
            ax_height.set_ylim(0, max(heights/1e3) * 1.1)
        ax_height.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_orbit_map(self, figsize=(8, 6), title='Satellite Orbital Decay Trajectory', title_fontsize=14,
                       tick_fontsize=10, path_color='red', start_marker_color='green',
                       end_marker_color='red', marker_size=10, scatter_size=50, cmap='viridis',
                       show_legend=True):
        """
        Plot the satellite orbit decay path as a standalone Cartopy map.
        
        Parameters:
        - figsize: tuple, figure size (width, height) in inches (default: (8, 6))
        - title: str, plot title (default: 'Satellite Orbit Decay Path')
        - title_fontsize: int, title font size (default: 14)
        - tick_fontsize: int, tick label font size (default: 10)
        - path_color: str, color of the orbit path (default: 'red')
        - start_marker_color: str, color of the start marker (default: 'green')
        - end_marker_color: str, color of the end marker (default: 'red')
        - marker_size: int, size of start/end markers (default: 10)
        - scatter_size: int, size of height scatter points (default: 50)
        - cmap: str, colormap for height scatter (default: 'viridis')
        - show_legend: bool, whether to show legend (default: True)
        """
        # Convert coordinates
        # lons, lats, heights, times, _ = self._convert_to_lonlat()
        lons = self.lon
        lats = self.lat
        heights = self.altitude

        # Create figure
        fig = plt.figure(figsize=figsize)
        ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Cartopy map
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax_map.gridlines(draw_labels=True, linestyle='--')
        ax_map.plot(lons, lats, color=path_color, linewidth=2, transform=ccrs.PlateCarree(), label='Orbit Path')
        ax_map.plot(lons[0], lats[0], 'o', color=start_marker_color, markersize=marker_size, 
                    transform=ccrs.PlateCarree(), label='Start')
        if len(lons) > 1:
            ax_map.plot(lons[-1], lats[-1], 'o', color=end_marker_color, markersize=marker_size, 
                        transform=ccrs.PlateCarree(), label='End')
        sc = ax_map.scatter(lons, lats, c=heights/1e3, cmap=cmap, transform=ccrs.PlateCarree(), 
                            s=scatter_size, alpha=0.6, label='Height (km)')
        plt.colorbar(sc, ax=ax_map, label='Height (km)')
        ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax_map.set_title(title, fontsize=title_fontsize)
        if show_legend:
            ax_map.legend()
        ax_map.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_crash_distribution(self, figsize=(8, 6), title='Predicted crash site distribution',
                                title_fontsize=14, label_fontsize=12, tick_fontsize=10,
                                box_color='blue', show_grid=True):
        """
        Plot the predicted crash longitude distribution as a standalone figure.
        
        Parameters:
        - figsize: tuple, figure size (width, height) in inches (default: (8, 6))
        - title: str, plot title (default: 'Predicted Crash Longitude Distribution')
        - title_fontsize: int, title font size (default: 14)
        - label_fontsize: int, axis label font size (default: 12)
        - tick_fontsize: int, tick label font size (default: 10)
        - box_color: str, color of boxplot (default: 'blue')
        - show_grid: bool, whether to show grid (default: True)
        """
        # Convert coordinates
        # lons, lats, heights, times, crash_lons_list = self._convert_to_lonlat()

        crash_lons_list = self.crash_lon_list
        lons = self.lon 
        lats = self.lat
        heights = self.altitude
        times = self.times

        forecasts = [crash_lons_list[id, :, 1] for id in range(0, crash_lons_list.shape[0])]
        labels = [str(id) for id in range(1, crash_lons_list.shape[0]+1)]

        # Create figure
        fig, ax_crash = plt.subplots(figsize=figsize)

        # Crash X Distribution
        if crash_lons_list is not None:
            # ax_crash.boxplot(crash_lons_list, positions=times, #widths=(times[1] - times[0])/2 if len(times) > 1 else 1,
            #                  patch_artist=True, boxprops=dict(facecolor=box_color, color=box_color))

            ax_crash.boxplot(forecasts, labels=labels, patch_artist=True, boxprops=dict(facecolor=box_color, color=box_color))

        ax_crash.set_xlabel('Forecast index', fontsize=label_fontsize)
        ax_crash.set_ylabel('Longitude (degrees)', fontsize=label_fontsize)
        ax_crash.set_title(title, fontsize=title_fontsize)

        if show_grid:
            ax_crash.grid(True)

        ax_crash.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax_crash.ticklabel_format(style='plain', axis='y')

        # if crash_lons_list is not None and len(times) > 0:
        #     # ax_crash.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
        #     #                  max(times) + 0.1 * (max(times) - min(times)))
        #     crash_lons_flat = np.concatenate(crash_lons_list)
        #     ax_crash.set_ylim(min(crash_lons_flat) * 0.9, max(crash_lons_flat) * 1.1)

        plt.tight_layout(pad=2.0)
        plt.show()

    def plot_orbit(self, figsize=(12, 10), title_fontsize=14, label_fontsize=12, tick_fontsize=10,
                   map_title='Satellite orbital decay trajectory', height_title='Altitude against time',
                   crash_title='Predicted crash site distribution', path_color='red',
                   height_line_color='blue', box_color='blue', show_grid=True, show_legend=True):
        """
        Plot the satellite orbit, height vs time, and crash x distribution in a 2x2 grid.
        
        Parameters:
        - figsize: tuple, figure size (width, height) in inches (default: (12, 10))
        - title_fontsize: int, title font size (default: 14)
        - label_fontsize: int, axis label font size (default: 12)
        - tick_fontsize: int, tick label font size (default: 10)
        - map_title: str, map subplot title (default: 'Satellite Orbit Decay Path')
        - height_title: str, height subplot title (default: 'Height vs Time')
        - crash_title: str, crash distribution subplot title (default: 'Predicted Crash Longitude Distribution')
        - path_color: str, color of the orbit path (default: 'red')
        - height_line_color: str, color of the height line (default: 'blue')
        - box_color: str, color of boxplot (default: 'blue')
        - show_grid: bool, whether to show grid (default: True)
        - show_legend: bool, whether to show legend (default: True)
        """
        # Convert coordinates
        # lons, lats, heights, times, crash_lons_list = self._convert_to_lonlat()
        crash_lons_list = self.crash_lon_list
        lons = self.lon 
        lats = self.lat
        heights = self.altitude
        times = self.times

        forecasts = [crash_lons_list[id, :, 1] for id in range(0, crash_lons_list.shape[0])]
        labels = [str(id) for id in range(1, crash_lons_list.shape[0]+1)]

        # Create figure with 2x2 grid
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, height_ratios=[2, 1])
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        ax_height = fig.add_subplot(gs[1, 0])
        ax_crash = fig.add_subplot(gs[1, 1])

        # Cartopy map
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax_map.gridlines(draw_labels=True, linestyle='--')
        ax_map.plot(lons, lats, color=path_color, linewidth=2, transform=ccrs.PlateCarree(), label='Orbit Path')
        ax_map.plot(lons[0], lats[0], 'go', markersize=10, transform=ccrs.PlateCarree(), label='Start')
        if len(lons) > 1:
            ax_map.plot(lons[-1], lats[-1], 'ro', markersize=10, transform=ccrs.PlateCarree(), label='End')
        sc = ax_map.scatter(lons, lats, c=heights/1e3, cmap='viridis', transform=ccrs.PlateCarree(), 
                            s=50, alpha=0.6, label='Altitude (km)')
        plt.colorbar(sc, ax=ax_map, label='Altitude (km)')
        ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax_map.set_title(map_title, fontsize=title_fontsize)
        if show_legend:
            ax_map.legend()

        # Height vs Time
        ax_height.plot(times, heights/1e3, marker='.', color=height_line_color, linewidth=2, label='Altitude (km)')
        ax_height.axhline(min(heights/1e3), label='Predictor termination', linestyle='--', color='red')
        ax_height.set_xlabel('Time (seconds)', fontsize=label_fontsize)
        ax_height.set_ylabel('Altitude (km)', fontsize=label_fontsize)
        ax_height.set_title(height_title, fontsize=title_fontsize)
        if show_grid:
            ax_height.grid(True)
        if show_legend:
            ax_height.legend()
        if len(heights) > 0 and len(times) > 0:
            ax_height.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
                              max(times) + 0.1 * (max(times) - min(times)))
            ax_height.set_ylim(0, max(heights/1e3) * 1.1)
            # ax_height.set_ylim(min(heights/1e3) * 0.9, max(heights/1e3) * 1.1)
        ax_height.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Crash X Distribution
        if crash_lons_list is not None:
            ax_crash.boxplot(forecasts, labels=labels, patch_artist=True, boxprops=dict(facecolor=box_color, color=box_color))

        ax_crash.set_xlabel('Forecast index', fontsize=label_fontsize)
        ax_crash.set_ylabel('Longitude (degrees)', fontsize=label_fontsize)
        ax_crash.set_title(crash_title, fontsize=title_fontsize)
        ax_crash.ticklabel_format(style='plain', axis='y')


        if show_grid:
            ax_crash.grid(True)

        ax_crash.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # if crash_lons_list is not None and len(times) > 0:
        #     ax_crash.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
        #                      max(times) + 0.1 * (max(times) - min(times)))
        #     crash_lons_flat = np.concatenate(crash_lons_list)
        #     ax_crash.set_ylim(min(crash_lons_flat) * 0.9, max(crash_lons_flat) * 1.1)

        plt.tight_layout(pad=2.0)
        plt.show()

    def save_plot(self, filename='orbit_decay.png', figsize=(12, 10), title_fontsize=14,
                  label_fontsize=12, tick_fontsize=10, map_title='Satellite Orbit Decay Path',
                  height_title='Height vs Time', crash_title='Predicted Crash Longitude Distribution',
                  path_color='red', height_line_color='blue', box_color='blue', show_grid=True,
                  show_legend=True, dpi=300, bbox_inches='tight'):
        """
        Save the orbit plot, height vs time, and crash x distribution in a 2x2 grid.
        
        Parameters:
        - filename: str, output file name (default: 'orbit_decay.png')
        - figsize: tuple, figure size (width, height) in inches (default: (12, 10))
        - title_fontsize: int, title font size (default: 14)
        - label_fontsize: int, axis label font size (default: 12)
        - tick_fontsize: int, tick label font size (default: 10)
        - map_title: str, map subplot title (default: 'Satellite Orbit Decay Path')
        - height_title: str, height subplot title (default: 'Height vs Time')
        - crash_title: str, crash distribution subplot title (default: 'Predicted Crash Longitude Distribution')
        - path_color: str, color of the orbit path (default: 'red')
        - height_line_color: str, color of the height line (default: 'blue')
        - box_color: str, color of boxplot (default: 'blue')
        - show_grid: bool, whether to show grid (default: True)
        - show_legend: bool, whether to show legend (default: True)
        - dpi: int, resolution for saved image (default: 300)
        - bbox_inches: str, bounding box for saving (default: 'tight')
        """
        # Convert coordinates
        lons, lats, heights, times, crash_lons_list = self._convert_to_lonlat()

        # Create figure with 2x2 grid
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, height_ratios=[2, 1])
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        ax_height = fig.add_subplot(gs[1, 0])
        ax_crash = fig.add_subplot(gs[1, 1])

        # Cartopy map
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax_map.gridlines(draw_labels=True, linestyle='--')
        ax_map.plot(lons, lats, color=path_color, linewidth=2, transform=ccrs.PlateCarree(), label='Orbit Path')
        ax_map.plot(lons[0], lats[0], 'go', markersize=10, transform=ccrs.PlateCarree(), label='Start')
        if len(lons) > 1:
            ax_map.plot(lons[-1], lats[-1], 'ro', markersize=10, transform=ccrs.PlateCarree(), label='End')
        sc = ax_map.scatter(lons, lats, c=heights/1e3, cmap='viridis', transform=ccrs.PlateCarree(), 
                            s=50, alpha=0.6, label='Height (km)')
        plt.colorbar(sc, ax=ax_map, label='Height (km)')
        ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax_map.set_title(map_title, fontsize=title_fontsize)
        if show_legend:
            ax_map.legend()

        # Height vs Time
        ax_height.plot(times, heights/1e3, color=height_line_color, linewidth=2, label='Height (km)')
        ax_height.set_xlabel('Time (seconds)', fontsize=label_fontsize)
        ax_height.set_ylabel('Height (km)', fontsize=label_fontsize)
        ax_height.set_title(height_title, fontsize=title_fontsize)
        if show_grid:
            ax_height.grid(True)
        if show_legend:
            ax_height.legend()
        if len(heights) > 0 and len(times) > 0:
            ax_height.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
                              max(times) + 0.1 * (max(times) - min(times)))
            ax_height.set_ylim(min(heights/1e3) * 0.9, max(heights/1e3) * 1.1)
        ax_height.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Crash X Distribution
        if crash_lons_list is not None:
            ax_crash.boxplot(crash_lons_list, positions=times, widths=(times[1] - times[0])/2 if len(times) > 1 else 1,
                             patch_artist=True, boxprops=dict(facecolor=box_color, color=box_color))
        ax_crash.set_xlabel('Time (seconds)', fontsize=label_fontsize)
        ax_crash.set_ylabel('Crash Longitude (degrees)', fontsize=label_fontsize)
        ax_crash.set_title(crash_title, fontsize=title_fontsize)
        if show_grid:
            ax_crash.grid(True)
        ax_crash.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        if crash_lons_list is not None and len(times) > 0:
            ax_crash.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
                             max(times) + 0.1 * (max(times) - min(times)))
            crash_lons_flat = np.concatenate(crash_lons_list)
            ax_crash.set_ylim(min(crash_lons_flat) * 0.9, max(crash_lons_flat) * 1.1)

        plt.tight_layout(pad=2.0)
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close()

    def animate_orbit(self, interval=50, figsize=(12, 12), title_fontsize=14, label_fontsize=12,
                     tick_fontsize=10, map_title='Satellite Orbital Decay Animation',
                     height_title='Altitude against time', crash_title='Predicted crash site distribution',
                     path_color='red', current_point_color='red', height_line_color='blue',
                     crash_point_color='black', crash_scatter_color='blue', marker_size=8,
                     scatter_size=50, cmap='viridis', show_grid=True, show_legend=True,
                     button_pos_play_pause=[0.35, 0.05, 0.1, 0.05], button_pos_reset=[0.5, 0.05, 0.1, 0.05]):
        """
        Animate the satellite orbit, height vs time, and crash x distribution in a 2x2 grid
        with play/pause and reset buttons.
        
        Parameters:
        - interval: float, time between frames in milliseconds (default: 50)
        - figsize: tuple, figure size (width, height) in inches (default: (12, 12))
        - title_fontsize: int, title font size (default: 14)
        - label_fontsize: int, axis label font size (default: 12)
        - tick_fontsize: int, tick label font size (default: 10)
        - map_title: str, map subplot title (default: 'Satellite Orbit Decay Animation')
        - height_title: str, height subplot title (default: 'Height vs Time')
        - crash_title: str, crash distribution subplot title (default: 'Predicted Crash Longitude Distribution')
        - path_color: str, color of the orbit path (default: 'red')
        - current_point_color: str, color of current position marker (default: 'red')
        - height_line_color: str, color of the height line (default: 'blue')
        - crash_point_color: str, color of crash points on map (default: 'black')
        - crash_scatter_color: str, color of crash scatter points (default: 'blue')
        - marker_size: int, size of markers (default: 8)
        - scatter_size: int, size of scatter points (default: 50)
        - cmap: str, colormap for height scatter (default: 'viridis')
        - show_grid: bool, whether to show grid (default: True)
        - show_legend: bool, whether to show legend (default: True)
        - button_pos_play_pause: list, position of play/pause button [x, y, width, height] (default: [0.35, 0.05, 0.1, 0.05])
        - button_pos_reset: list, position of reset button [x, y, width, height] (default: [0.5, 0.05, 0.1, 0.05])
        """
        # Convert coordinates
        # lons, lats, heights, times, crash_lons_list = self._convert_to_lonlat()

        crash_lons_list = self.crash_lon_list
        lons = self.lon 
        lats = self.lat
        heights = self.altitude
        times = self.times

        # Create figure with 2x2 grid and extra space for buttons
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, height_ratios=[2, 1], bottom=0.15)
        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        ax_height = fig.add_subplot(gs[1, 0])
        ax_crash = fig.add_subplot(gs[1, 1])

        # Cartopy map
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax_map.gridlines(draw_labels=True, linestyle='--')
        ax_map.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax_map.set_title(map_title, fontsize=title_fontsize)
        line, = ax_map.plot([], [], color=path_color, linewidth=2, transform=ccrs.PlateCarree(), label='Orbit Path')
        point, = ax_map.plot([], [], 'o', color=current_point_color, markersize=marker_size, 
                            transform=ccrs.PlateCarree(), label='Current Position')
        ax_map.plot(lons[0], lats[0], 'o', color='green', markersize=marker_size, 
                    transform=ccrs.PlateCarree(), label='Start')
        sc = ax_map.scatter([lons[0]], [lats[0]], c=[heights[0]/1e3], cmap=cmap, 
                            transform=ccrs.PlateCarree(), s=scatter_size, alpha=0.6, label='Height (km)')
        crash_point = ax_map.scatter([], [], c=crash_point_color, marker='x', s=scatter_size, alpha=0.5, 
                                    transform=ccrs.PlateCarree(), label='Mean predicted crash site')
        plt.colorbar(sc, ax=ax_map, label='Altitude (km)')
        if show_legend:
            ax_map.legend()

        # Height vs Time
        ax_height.set_xlabel('Time (seconds)', fontsize=label_fontsize)
        ax_height.set_ylabel('Altitude (km)', fontsize=label_fontsize)
        ax_height.set_title(height_title, fontsize=title_fontsize)
        if show_grid:
            ax_height.grid(True)
        height_line, = ax_height.plot([], [], color=height_line_color, linewidth=2, label='Altitude (km)')
        if show_legend:
            ax_height.legend()
        if len(heights) > 0 and len(times) > 0:
            ax_height.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
                              max(times) + 0.1 * (max(times) - min(times)))
            ax_height.set_ylim(min(heights/1e3) * 0.9, max(heights/1e3) * 1.1)
        ax_height.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Crash X Distribution
        ax_crash.set_xlabel('Time (seconds)', fontsize=label_fontsize)
        ax_crash.set_ylabel('Longitude (degrees)', fontsize=label_fontsize)
        ax_crash.set_title(crash_title, fontsize=title_fontsize)
        if show_grid:
            ax_crash.grid(True)
        if crash_lons_list is not None and len(times) > 0:
            ax_crash.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
                             max(times) + 0.1 * (max(times) - min(times)))
            crash_lons_flat = np.concatenate(crash_lons_list)
            ax_crash.set_ylim(min(crash_lons_flat) * 0.9, max(crash_lons_flat) * 1.1)
        ax_crash.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        crash_scatter = None

        # Animation state
        is_running = [True]  # Track if animation is running

        def init():
            line.set_data([], [])
            point.set_data([], [])
            sc.set_offsets(np.c_[lons[:1], lats[:1]])
            sc.set_array(np.array([heights[0]/1e3]))
            crash_point.set_offsets(np.empty((0, 2)))
            height_line.set_data([], [])
            if crash_lons_list is not None:
                ax_crash.clear()
                ax_crash.set_xlabel('Time (seconds)', fontsize=label_fontsize)
                ax_crash.set_ylabel('Crash Longitude (degrees)', fontsize=label_fontsize)
                ax_crash.set_title(crash_title, fontsize=title_fontsize)
                if show_grid:
                    ax_crash.grid(True)
                ax_crash.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
                                 max(times) + 0.1 * (max(times) - min(times)))
                ax_crash.set_ylim(min(crash_lons_flat) * 0.9, max(crash_lons_flat) * 1.1)
                ax_crash.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            return [line, point, sc, crash_point, height_line] + ([crash_scatter] if crash_scatter is not None else [])

        def update(frame):
            # Update map
            line.set_data(lons[:frame+1], lats[:frame+1])
            point.set_data([lons[frame]], [lats[frame]])
            sc.set_offsets(np.c_[lons[:frame+1], lats[:frame+1]])
            sc.set_array(heights[:frame+1]/1e3)
            if crash_lons_list is not None:
                crash_lons = crash_lons_list[frame]
                crash_lats = np.zeros_like(crash_lons)
                crash_point.set_offsets(np.c_[crash_lons, crash_lats])
            # Update height vs time
            height_line.set_data(times[:frame+1], heights[:frame+1]/1e3)
            # Update crash x distribution
            if crash_lons_list is not None:
                ax_crash.clear()
                ax_crash.set_xlabel('Time (seconds)', fontsize=label_fontsize)
                ax_crash.set_ylabel('Crash Longitude (degrees)', fontsize=label_fontsize)
                ax_crash.set_title(crash_title, fontsize=title_fontsize)
                if show_grid:
                    ax_crash.grid(True)
                ax_crash.set_xlim(min(times) - 0.1 * (max(times) - min(times)), 
                                 max(times) + 0.1 * (max(times) - min(times)))
                ax_crash.set_ylim(min(crash_lons_flat) * 0.9, max(crash_lons_flat) * 1.1)
                ax_crash.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                nonlocal crash_scatter
                crash_scatter = ax_crash.scatter([times[frame]] * len(crash_lons_list[frame]), 
                                                crash_lons_list[frame], c=crash_scatter_color, 
                                                marker='x', s=scatter_size, alpha=0.5)
            return [line, point, sc, crash_point, height_line] + ([crash_scatter] if crash_scatter is not None else [])

        # Create animation
        ani = FuncAnimation(fig, update, init_func=init, frames=len(lons), interval=interval, 
                            blit=False, repeat=False)

        # Add buttons
        # ax_play_pause = plt.axes(button_pos_play_pause)
        # ax_reset = plt.axes(button_pos_reset)
        # btn_play_pause = Button(ax_play_pause, 'Pause')
        # btn_reset = Button(ax_reset, 'Reset')

        # def toggle_play_pause(event):
        #     if is_running[0]:
        #         ani.event_source.stop()
        #         is_running[0] = False
        #         btn_play_pause.label.set_text('Play')
        #     else:
        #         ani.event_source.start()
        #         is_running[0] = True
        #         btn_play_pause.label.set_text('Pause')
        #     fig.canvas.draw()

        # def reset(event):
        #     ani.frame_seq = ani.new_frame_seq()
        #     if ani.event_source is not None:
        #         ani.event_source.stop()
        #     ani._iter_gen = iter(range(len(lons)))
        #     is_running[0] = False
        #     btn_play_pause.label.set_text('Play')
        #     init()
        #     ani._draw_frame(0)
        #     fig.canvas.draw()

        # btn_play_pause.on_clicked(toggle_play_pause)
        # btn_reset.on_clicked(reset)

        # Adjust layout manually
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95, hspace=0.3, wspace=0.2)

        plt.show()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictor_forecasts_path = os.path.join(current_dir, "test_data/predictor_forecasts.npz")
    predictor_posterior_trajectory_path = os.path.join(current_dir, "test_data/predictor_posterior_trajectory.npz")

    predictor_forecasts = np.load(predictor_forecasts_path)
    predictor_posterior_trajectory = np.load(predictor_posterior_trajectory_path)
    print(predictor_posterior_trajectory)

    xs = predictor_posterior_trajectory['xs'][:25]
    ys = predictor_posterior_trajectory['ys'][:25]
    vxs = predictor_posterior_trajectory['vx'][:25]
    vys = predictor_posterior_trajectory['vy'][:25]

    forecast_states = predictor_forecasts['states']
    forecast_states_repeated = np.repeat(forecast_states, repeats=5, axis=0)
    n_steps = forecast_states_repeated.shape[0]
    interval_seconds = 5
    times = np.arange(0, n_steps * interval_seconds, interval_seconds)

    crash_x_list = [forecast_states_repeated[i, :, 0] for i in range(n_steps)]

    vis = Visualiser(times, xs, ys, vxs, vys, crash_x_list)
    # vis.plot_orbit()
    # vis.save_plot('orbit_decay.png')
    # vis.animate_orbit(interval=50)

    vis.plot_orbit_map(path_color='blue', cmap='magma')
    vis.animate_orbit(current_point_color='yellow', interval=100)