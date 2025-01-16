import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings

class CriticalSpeedAnalyzer:
    """A class to analyze running performance using Critical Speed methodology with 5 training zones."""
    
    def __init__(self):
        self.model = LinearRegression()
        self.cs_m_per_s = None
        self.d_prime = None
        self.r2_score = None
        
    @staticmethod
    def time_to_seconds(time_str: str) -> float:
        """Convert time string (min:sec or decimal) to seconds with error handling."""
        try:
            # If the user enters something without a colon, assume it's already just seconds
            if ':' not in time_str:
                return float(time_str)
            
            # Otherwise split by ':'
            minutes, seconds = map(float, time_str.split(':'))
            if seconds >= 60:
                raise ValueError("Seconds should be less than 60")
            return minutes * 60 + seconds
        except ValueError as e:
            raise ValueError(f"Invalid time format: {time_str}. Use 'min:sec' or seconds.") from e

    @staticmethod
    def format_pace(pace_min_km: float) -> str:
        """Format pace as min:sec per km with proper rounding."""
        minutes = int(pace_min_km)
        seconds = round((pace_min_km - minutes) * 60)
        if seconds == 60:
            minutes += 1
            seconds = 0
        return f"{minutes}:{seconds:02d}"

    @staticmethod
    def format_time_h_m(seconds: float) -> str:
        """Format time in hours and minutes."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}:{minutes:02d}"

    def calculate_critical_speed(
        self, 
        distances,       # List of floats
        race_times,      # List of strings
        distance_unit='meters'
    ) -> dict:
        """
        Calculate Critical Speed and related metrics.
        
        Parameters:
        -----------
        distances : list[float]
            List of race distances (e.g., [400, 800, 5000])
        race_times : list[str]
            List of race times in 'min:sec' or decimal format
        distance_unit : str
            'meters' or 'kilometers'
            
        Returns:
        --------
        dict : Dictionary containing analysis results
        """
        if len(distances) != len(race_times):
            raise ValueError("Number of distances must match number of times")
        if len(distances) < 3:
            warnings.warn("At least 3 data points recommended for reliable results")
            
        # Convert distances to meters if needed
        conversion = 1000 if distance_unit.lower() == 'kilometers' else 1
        distances_m = np.array(distances) * conversion
        
        # Convert times to seconds and calculate speeds
        times_seconds = np.array([self.time_to_seconds(t) for t in race_times])
        speeds = distances_m / times_seconds
        
        # Prepare data for regression
        X = (1 / distances_m).reshape(-1, 1)
        y = speeds
        
        # Fit the linear model
        self.model.fit(X, y)
        self.cs_m_per_s = self.model.intercept_
        self.d_prime = self.model.coef_[0]
        self.r2_score = r2_score(y, self.model.predict(X))
        
        # Calculate CS in various units
        cs_min_per_km = 1000 / self.cs_m_per_s / 60  # min/km
        cs_km_per_h = self.cs_m_per_s * 3.6          # km/h
        
        # Training zone definitions (percent of CS)
        zone_percentages = {
            'Zone 5 (Repetition - 115% - 150%)': (1.50, 1.15),
            'Zone 4 (Interval - 100% - 115%)':    (1.15, 1.00),
            'Zone 3 (Threshold - 90% - 100%)':    (1.00, 0.90),
            'Zone 2 (Moderate - 80% - 90%)':      (0.90, 0.80),
            'Zone 1 (Easy65-80%)':                (0.80, 0.65)
        }
        
        training_paces = {}
        for zone, (upper, lower) in zone_percentages.items():
            upper_speed = self.cs_m_per_s * upper
            lower_speed = self.cs_m_per_s * lower
            upper_pace = 1000 / upper_speed / 60
            lower_pace = 1000 / lower_speed / 60
            
            training_paces[zone] = {
                'range_kmh': (round(lower_speed * 3.6, 1), round(upper_speed * 3.6, 1)),
                'range_pace': (self.format_pace(lower_pace), self.format_pace(upper_pace))
            }
        
        # Marathon prediction at 85% of CS
        marathon_speed = self.cs_m_per_s * 0.85
        marathon_pace_min_km = 1000 / marathon_speed / 60
        marathon_time_seconds = 42195 / marathon_speed
        marathon_time = self.format_time_h_m(marathon_time_seconds)

        # Half Marathon prediction at 94% of CS
        half_marathon_speed = self.cs_m_per_s * 0.94
        half_marathon_pace_min_km = 1000 / half_marathon_speed / 60
        half_marathon_time_seconds = 21097.5 / half_marathon_speed
        half_marathon_time = self.format_time_h_m(half_marathon_time_seconds)
        
        return {
            'critical_speed_ms': round(self.cs_m_per_s, 2),
            'critical_speed_kmh': round(cs_km_per_h, 2),
            'critical_pace_minkm': self.format_pace(cs_min_per_km),
            'd_prime': round(self.d_prime, 2),
            'r2_score': round(self.r2_score, 4),
            'training_zones': training_paces,
            'marathon_prediction': {
                'pace': self.format_pace(marathon_pace_min_km),
                'time': marathon_time
            },
            'half_marathon_prediction': {
                'pace': self.format_pace(half_marathon_pace_min_km),
                'time': half_marathon_time
            },
            'race_data': pd.DataFrame({
                'Distance': distances_m,
                'Time': race_times,
                'Speed (m/s)': np.round(speeds, 2)
            })
        }

    def plot_analysis(self, results: dict, save_path: str = None):
        """
        Create a visualization focused on the Training Zones.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object for Streamlit to display via st.pyplot()
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Training Zones as horizontal ranges
        zone_colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#DDA0DD']
        zones = list(results['training_zones'].keys())
        
        y_pos = np.arange(len(zones))

        for i, (zone, color) in enumerate(zip(zones, zone_colors)):
            zone_data = results['training_zones'][zone]
            # Convert pace strings like '5:00' to minutes
            pace_range = [self.time_to_seconds(p)/60 for p in zone_data['range_pace']]
            
            ax.barh(
                y_pos[i], 
                pace_range[0] - pace_range[1], 
                left=pace_range[1], 
                color=color, 
                alpha=0.6
            )
            
            # Add text labels
            ax.text(
                pace_range[1], 
                y_pos[i], 
                f' {zone_data["range_pace"][1]} - {zone_data["range_pace"][0]} min/km\n'
                f' ({zone_data["range_kmh"][0]} - {zone_data["range_kmh"][1]} km/h)', 
                va='center', 
                fontsize=9
            )
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(zones)
        ax.set_title('Training Zones', fontsize=12, pad=20)
        ax.set_xlabel('Minutes per Kilometer', fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add model statistics
        stats_text = (
            f"R² Score: {results['r2_score']}\n"
            f"D': {results['d_prime']} m"
        )
        fig.text(
            0.98, 0.02, 
            stats_text, 
            ha='right', 
            va='bottom', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )
        
        plt.tight_layout()
        
        # Optionally save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Return the figure to Streamlit
        return fig


def main():
    """Streamlit app main function."""
    st.title("Critical Speed Analysis")
    st.write("Enter your 400m, 800m, and 5000m times. Use the'mm:ss' format e.g. 1:41")

    # Inputs for times
    time_400 = st.text_input("400m Time", value="1:30")
    time_800 = st.text_input("800m Time", value="3:00")
    time_5000 = st.text_input("5000m Time", value="20:00")

    # Distances in meters
    distances = [400, 800, 5000]
    race_times = [time_400, time_800, time_5000]

    # Button to run analysis
    if st.button("Analyze"):
        analyzer = CriticalSpeedAnalyzer()
        
        try:
            # Calculate critical speed
            results = analyzer.calculate_critical_speed(distances, race_times)
            
            # Display results
            st.subheader("Results Summary")
            st.write(f"**Critical Pace:** {results['critical_pace_minkm']} min/km")
            #st.write(f"**Critical Speed:** {results['critical_speed_kmh']} km/h")
            #st.write(f"**D' (meters):** {results['d_prime']}")
            st.write(f"**R² Score:** {results['r2_score']}")
            
            # Training Zones
            st.subheader("Training Zones")
            for zone, data in results['training_zones'].items():
                st.markdown(f"**{zone}**")
                st.write(f"- Pace Range: {data['range_pace'][1]} - {data['range_pace'][0]} min/km")
                #st.write(f"- Speed Range: {data['range_kmh'][0]} - {data['range_kmh'][1]} km/h")
            
            # Marathon prediction
            st.subheader("Marathon Prediction")
            st.write(f"- Pace: {results['marathon_prediction']['pace']} min/km")
            st.write(f"- Finish Time: {results['marathon_prediction']['time']}")
            
            # Half Marathon prediction
            st.subheader("Half Marathon Prediction")
            st.write(f"- Pace: {results['half_marathon_prediction']['pace']} min/km")
            st.write(f"- Finish Time: {results['half_marathon_prediction']['time']}")
            
            # Display race data in a table
            st.subheader("Race Data")
            st.dataframe(results['race_data'])
            
            # Plot the training zones
            fig = analyzer.plot_analysis(results)
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")


# If you prefer, you can still keep this, but it's not strictly necessary in Streamlit:
if __name__ == "__main__":
    main()
