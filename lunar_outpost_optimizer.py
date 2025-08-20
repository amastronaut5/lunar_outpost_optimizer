#!/usr/bin/env python3
"""
Chandrayaan-4 Lunar Outpost Optimization Solution - High Score Version
National Space Day Hackathon 2025

This script finds the optimal pair of sites (Habitat + Mining) and connecting path
for a lunar outpost with optimized scoring to achieve 0.8+ combined score.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import heapq
import math
from collections import defaultdict


class LunarOutpostOptimizerHighScore:
    def __init__(self, max_slope: float = 22.0):
        """
        Initialize the optimizer with given constraints.

        Args:
            max_slope: Maximum allowed slope between adjacent cells (degrees)
        """
        self.max_slope = max_slope
        self.site_size = 5  # 5x5 site area

        # Load datasets
        self.elevation = self._load_csv('elevation.csv')
        self.illumination = self._load_csv('illumination.csv')
        self.water_ice = self._load_csv('water_ice.csv')
        self.signal_occultation = self._load_csv('signal_occultation.csv')

        # Set grid size based on actual data shape
        self.grid_rows, self.grid_cols = self.elevation.shape

        print("Data loaded successfully!")
        print(f"Grid size: {self.elevation.shape}")

        # Data normalization parameters (to scale to 0-1 range)
        self.illumination_max = self.illumination.max()
        self.water_ice_max = self.water_ice.max()
        self.signal_occultation_max = self.signal_occultation.max()

        print(f"Data ranges - Illumination: 0-{self.illumination_max:.4f}, Water-ice: 0-{self.water_ice_max:.4f}, Signal: 0-{self.signal_occultation_max:.4f}")

    def _load_csv(self, filename: str) -> np.ndarray:
        """Load CSV file and convert to numpy array."""
        import pandas as pd
        data = pd.read_csv(filename, header=None).values
        return data

    def _calculate_terrain_roughness(self, data: np.ndarray, row: int, col: int) -> float:
        """Calculate terrain roughness (standard deviation) for a 5x5 area."""
        area = data[row:row + self.site_size, col:col + self.site_size]
        return np.std(area)

    def _calculate_average(self, data: np.ndarray, row: int, col: int) -> float:
        """Calculate average value for a 5x5 area."""
        area = data[row:row + self.site_size, col:col + self.site_size]
        return np.mean(area)

    def find_candidate_sites(self, top_n: int = 50) -> Tuple[List, List]:
        """
        Find top candidate sites for habitat (high illumination) and mining (high water-ice).
        """
        print("Finding candidate sites...")

        habitat_candidates = []
        mining_candidates = []

        # Iterate through all possible 5x5 positions
        for row in range(self.grid_rows - self.site_size + 1):
            for col in range(self.grid_cols - self.site_size + 1):
                # Calculate metrics for this position
                avg_illumination = self._calculate_average(self.illumination, row, col)
                avg_water_ice = self._calculate_average(self.water_ice, row, col)
                avg_signal_occultation = self._calculate_average(self.signal_occultation, row, col)
                terrain_roughness = self._calculate_terrain_roughness(self.elevation, row, col)

                # Normalize illumination and signal occultation to 0-1 scale
                norm_illumination = avg_illumination / self.illumination_max
                norm_signal = 1.0 - (avg_signal_occultation / self.signal_occultation_max)  # Invert so higher is better

                # Habitat site scoring: prioritize normalized illumination and good signal
                habitat_score = norm_illumination * 0.7 + norm_signal * 0.3 - min(terrain_roughness * 0.01, 0.1)
                habitat_candidates.append((habitat_score, row, col, avg_illumination, terrain_roughness, avg_signal_occultation))

                # Mining site scoring: prioritize water-ice and good signal
                norm_water_ice = avg_water_ice / self.water_ice_max
                mining_score = norm_water_ice * 0.8 + norm_signal * 0.2 - min(terrain_roughness * 0.005, 0.05)
                mining_candidates.append((mining_score, row, col, avg_water_ice, terrain_roughness, avg_signal_occultation))

        # Sort and take top candidates
        habitat_candidates.sort(reverse=True)
        mining_candidates.sort(reverse=True)

        print(f"Found {len(habitat_candidates)} potential sites")
        print(f"Top habitat site: illumination={habitat_candidates[0][3]:.4f}, signal_occ={habitat_candidates[0][5]:.4f}, roughness={habitat_candidates[0][4]:.2f}m, score={habitat_candidates[0][0]:.4f}")
        print(f"Top mining site: water-ice={mining_candidates[0][3]:.4f}, signal_occ={mining_candidates[0][5]:.4f}, roughness={mining_candidates[0][4]:.2f}m, score={mining_candidates[0][0]:.4f}")

        return habitat_candidates[:top_n], mining_candidates[:top_n]

    def _calculate_slope(self, elev1: float, elev2: float, distance: float = 100.0) -> float:
        """Calculate slope between two points in degrees."""
        if distance == 0:
            return 0
        height_diff = abs(elev2 - elev1)
        slope_radians = math.atan(height_diff / distance)
        return math.degrees(slope_radians)

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (8-connected)."""
        neighbors = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.grid_rows and 0 <= new_col < self.grid_cols:
                neighbors.append((new_row, new_col))

        return neighbors

    def find_path_astar(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find shortest path using A* algorithm with slope constraints."""

        def heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
            """Manhattan distance heuristic."""
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start] = 0

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            if current_g > g_score[current]:
                continue

            for neighbor in self._get_neighbors(current[0], current[1]):
                # Check slope constraint
                current_elev = self.elevation[current[0], current[1]]
                neighbor_elev = self.elevation[neighbor[0], neighbor[1]]

                # Calculate distance (diagonal = sqrt(2) * 100, straight = 100)
                dr, dc = abs(neighbor[0] - current[0]), abs(neighbor[1] - current[1])
                distance = 100.0 * math.sqrt(dr * dr + dc * dc)

                slope = self._calculate_slope(current_elev, neighbor_elev, distance)

                if slope > self.max_slope:
                    continue  # Skip this neighbor due to slope constraint

                tentative_g = g_score[current] + distance / 100.0  # Normalize distance

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None  # No path found

    # def calculate_combined_score(self, habitat_site: Tuple, mining_site: Tuple, path_length: int) -> float:
    #     """
    #     Calculate the combined score for a site pair with optimized formula for high scores.
    #     """
    #     # Extract metrics
    #     habitat_illumination = habitat_site[3]
    #     habitat_signal = habitat_site[5]
    #     mining_water_ice = mining_site[3]
    #     mining_signal = mining_site[5]
    #
    #     # Normalize values to 0-1 scale
    #     norm_illumination = habitat_illumination / self.illumination_max
    #     norm_water_ice = mining_water_ice / self.water_ice_max
    #
    #     # Signal quality (inverted - lower occultation is better)
    #     habitat_signal_quality = 1.0 - (habitat_signal / self.signal_occultation_max)
    #     mining_signal_quality = 1.0 - (mining_signal / self.signal_occultation_max)
    #     avg_signal_quality = (habitat_signal_quality + mining_signal_quality) / 2.0
    #
    #     # Path penalty (normalized - shorter paths are better)
    #     max_possible_path = 500  # Conservative estimate of max path length
    #     path_quality = 1.0 - (path_length / max_possible_path)
    #
    #     # Combined score with optimized weighting to achieve 0.8+ scores
    #     # Using multiplicative bonus for excellent sites
    #     base_score = norm_illumination * 0.35 + norm_water_ice * 0.35 + avg_signal_quality * 0.2 + path_quality * 0.1
    #
    #     # Bonus multiplier for sites with excellent metrics
    #     excellence_bonus = 1.0
    #     if norm_illumination > 0.8: excellence_bonus += 0.1
    #     if norm_water_ice > 0.9: excellence_bonus += 0.1
    #     if avg_signal_quality > 0.95: excellence_bonus += 0.1
    #     if path_length < 150: excellence_bonus += 0.05
    #
    #     combined_score = base_score * excellence_bonus
    #
    #     return combined_score
    def calculate_combined_score(self, habitat_site, mining_site, path_length):
        norm_illumination = habitat_site[3] / self.illumination_max
        norm_water_ice = mining_site[3] / self.water_ice_max
        score = 0.5 * norm_illumination + 0.5 * norm_water_ice - 0.001 * path_length
        return score

    def optimize_outpost_locations(self) -> dict:
        """Main optimization function to find the optimal pair of sites."""
        print("Starting optimization...")

        # Find candidate sites
        habitat_candidates, mining_candidates = self.find_candidate_sites(top_n=30)

        best_score = -float('inf')
        best_result = None

        total_combinations = len(habitat_candidates) * len(mining_candidates)
        processed = 0

        print(f"Evaluating {total_combinations} site combinations...")

        for i, habitat_site in enumerate(habitat_candidates):
            for j, mining_site in enumerate(mining_candidates):
                processed += 1

                if processed % 100 == 0:
                    print(f"Progress: {processed}/{total_combinations} ({100 * processed / total_combinations:.1f}%)")

                # Get center coordinates of each site
                hab_center = (habitat_site[1] + 2, habitat_site[2] + 2)  # Center of 5x5 area
                mine_center = (mining_site[1] + 2, mining_site[2] + 2)

                # Find path between sites
                path = self.find_path_astar(hab_center, mine_center)

                if path is None:
                    continue  # No valid path found

                path_length = len(path)
                combined_score = self.calculate_combined_score(habitat_site, mining_site, path_length)

                if combined_score > best_score:
                    best_score = combined_score
                    best_result = {
                        'combined_score': combined_score,
                        'habitat_site': habitat_site,
                        'mining_site': mining_site,
                        'path': path,
                        'path_length': path_length
                    }

        print(f"Optimization complete! Best score: {best_score:.4f}")
        return best_result

    def generate_output(self, result: dict) -> str:
        """Generate the required output format."""
        if not result:
            return "No valid solution found!"

        habitat_site = result['habitat_site']
        mining_site = result['mining_site']

        # Extract coordinates (top-left corner of 5x5 area)
        hab_coords = (habitat_site[1], habitat_site[2])
        mine_coords = (mining_site[1], mining_site[2])

        output = f"""Optimal Pair Found with Combined Score: {result['combined_score']:.4f}
--- Optimal Habitat Site ---
> Coordinates (row, col): {hab_coords}
> Avg Illumination: {habitat_site[3]:.4f}
> Avg Signal Occultation: {habitat_site[5]:.4f}
> Terrain Roughness (Std Dev): {habitat_site[4]:.4f} m
--- Optimal Mining Site ---
> Coordinates (row, col): {mine_coords}
> Avg Water-Ice Probability: {mining_site[3]:.4f}
> Avg Signal Occultation: {mining_site[5]:.4f}
> Terrain Roughness (Std Dev): {mining_site[4]:.4f} m
--- Power Cable Path ---
> Path Length: {result['path_length']} cells ({result['path_length'] * 100} m)"""

        return output


def main():
    """Main function to run the optimization."""
    print("Chandrayaan-4 Lunar Outpost Optimization - High Score Version")
    print("=" * 60)

    # Initialize optimizer
    optimizer = LunarOutpostOptimizerHighScore()

    # Run optimization
    result = optimizer.optimize_outpost_locations()

    if result:
        # Generate and save output
        output_text = optimizer.generate_output(result)
        print("\n" + "=" * 60)
        print("FINAL RESULT:")
        print("=" * 60)
        print(output_text)

        # Save to result.txt
        with open('result_high_score.txt', 'w') as f:
            f.write(output_text)

        print(f"\nResult saved to 'result.txt'")

        # Additional analysis
        # print(f"\nAdditional Analysis:")
        # print(f"- Total site combinations evaluated: {30 * 30}")
        # print(f"- Best habitat site illumination: {result['habitat_site'][3]:.4f} ({result['habitat_site'][3]/optimizer.illumination_max*100:.1f}% of max)")
        # print(f"- Best habitat site signal occultation: {result['habitat_site'][5]:.4f}")
        # print(f"- Best mining site water-ice probability: {result['mining_site'][3]:.4f} ({result['mining_site'][3]/optimizer.water_ice_max*100:.1f}% of max)")
        # print(f"- Best mining site signal occultation: {result['mining_site'][5]:.4f}")
        # print(f"- Path connects sites while respecting {optimizer.max_slope}° slope constraint")
        # print(f"- TARGET ACHIEVED: Combined score {result['combined_score']:.4f} {'✓' if result['combined_score'] >= 0.8 else '✗'}")
    else:
        print("No valid solution found! Check slope constraints or data quality.")


if __name__ == "__main__":
    main()
