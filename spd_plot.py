import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# Ensure system libraries are correctly recognized
print(sys.path)
sys.path.append("/usr/lib/python3/dist-packages/")

def read_light_spd(filename):
    return pd.read_csv(filename)

def read_frame_statistics_spd(filename):
    return pd.read_csv(filename)

def calculate_log_derivative(df):
    df['LogIntensity'] = np.log(df['Intensity'])
    df['LogIntensityDerivative'] = np.gradient(df['LogIntensity'])
    return df

def calculate_dominant_event(pos_count, neg_count, zero_count, pos_factor, neg_factor):
    all_counts = pos_count + neg_count + zero_count
    if all_counts == 0:
        return 0
    dominant_event = (pos_count * pos_factor - neg_count * neg_factor) / all_counts
    return dominant_event

def calculate_cumsum(df):
    return df['Dominant Event'].cumsum()

def calculate_exp(cumsum, exp_factor):
    return np.exp(cumsum / exp_factor)

def process_frame_statistics_spd(df, pos_factor, neg_factor, exp_factor):
    df['Dominant Event'] = df.apply(lambda row: calculate_dominant_event(row['PosCount'], row['NegCount'], row['ZeroCount'], pos_factor, neg_factor), axis=1)
    df['Cumsum'] = calculate_cumsum(df)
    df['Cumsum'] = df['Cumsum'] - df['Cumsum'].max()
    df['EXP'] = calculate_exp(df['Cumsum'], exp_factor)
    return df

def rescale_series(series):
    return 2 * (series - series.min()) / (series.max() - series.min()) - 1

def plot_data(light_spd_df, frame_statistics_spd_df, neg_factor, pos_factor, exp_factor, output_file):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Plot log intensity derivative
    axs[0, 0].set_title('Log Intensity Derivative')
    axs[0, 0].plot(light_spd_df['Wavelength'], light_spd_df['LogIntensityDerivative'], color='tab:blue')
    axs[0, 0].axhline(0, color='red', linestyle='dotted')

    # Plot cumulative sum of dominant events
    axs[1, 1].set_title('Cumulative Sum of Dominant Events')
    axs[1, 1].plot(frame_statistics_spd_df.index, frame_statistics_spd_df['Cumsum'], color='tab:orange')
    axs[1, 1].axhline(0, color='red', linestyle='dotted')

    # Plot log intensity
    axs[0, 1].set_title('Log Intensity')
    axs[0, 1].plot(light_spd_df['Wavelength'], light_spd_df['LogIntensity'], color='tab:green')
    axs[0, 1].axhline(0, color='red', linestyle='dotted')

    # Plot dominant event rescaled
    rescaled_dominant_event = rescale_series(frame_statistics_spd_df['Dominant Event'])
    axs[1, 0].set_title('Dominant Event (Rescaled)')
    axs[1, 0].plot(frame_statistics_spd_df.index, rescaled_dominant_event, color='tab:purple')
    axs[1, 0].axhline(0, color='red', linestyle='dotted')

    # Plot original SPD
    axs[0, 2].set_title('Original SPD')
    axs[0, 2].plot(light_spd_df['Wavelength'], light_spd_df['Intensity'], color='tab:brown')
    axs[0, 2].axhline(0, color='red', linestyle='dotted')

    # Plot exp of cumsum
    axs[1, 2].set_title('EXP of Cumsum')
    axs[1, 2].plot(frame_statistics_spd_df.index, frame_statistics_spd_df['EXP'], color='tab:cyan')
    axs[1, 2].axhline(0, color='red', linestyle='dotted')

    # Annotate factors on the plot
    annotation_text = f'neg_factor = {neg_factor:.2f}\npos_factor = {pos_factor:.2f}\nexp_factor = {exp_factor:.2f}'
    fig.text(0.5, 0.95, annotation_text, ha='center', va='center', fontsize=12, color='black')

    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def main(light_spd_file, frame_statistics_spd_file, output_file, neg_factor, pos_factor, exp_factor):
    light_spd_df = read_light_spd(light_spd_file)
    frame_statistics_spd_df = read_frame_statistics_spd(frame_statistics_spd_file)

    if neg_factor is None or pos_factor is None:
        max_neg = frame_statistics_spd_df['NegCount'].max()
        max_pos = frame_statistics_spd_df['PosCount'].max()
        sum_counts = frame_statistics_spd_df.iloc[0][['NegCount', 'ZeroCount', 'PosCount']].sum()
        
        if neg_factor is None:
            neg_factor = sum_counts / max_neg
        if pos_factor is None:
            pos_factor = sum_counts / max_pos

    print(f"Calculated neg_factor: {neg_factor:.2f}")
    print(f"Calculated pos_factor: {pos_factor:.2f}")

    light_spd_df = calculate_log_derivative(light_spd_df)
    frame_statistics_spd_df = process_frame_statistics_spd(frame_statistics_spd_df, pos_factor, neg_factor, exp_factor)
    
    # Initial exp_factor
    exp_factor = abs(frame_statistics_spd_df["Cumsum"].min())
    frame_statistics_spd_df['EXP'] = calculate_exp(frame_statistics_spd_df['Cumsum'], exp_factor)

    # Rescale EXP to 0 and 1
    frame_statistics_spd_df['EXP'] = (rescale_series(frame_statistics_spd_df['EXP'])+1)/2


    frame_statistics_spd_df.to_csv('frame_statistics_spd_completed.csv', index=False)

    print("Completed CSV saved as frame_statistics_spd_completed.csv")
    print(f"Calculated exp_factor: {exp_factor:.2f}")

    plot_data(light_spd_df, frame_statistics_spd_df, neg_factor, pos_factor, exp_factor, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Complete the frame_statistics_spd.csv with true values, rescale the dominant event, and plot the specified data in a 2x3 grid.')
    parser.add_argument('--light_spd_file', type=str, required=True, help='Path to the light_spd.csv file')
    parser.add_argument('--frame_statistics_spd_file', type=str, required=True, help='Path to the frame_statistics_spd.csv file')
    parser.add_argument('--output_file', type=str, default='log_derivative_dominant_event.png', help='Output file for the plot')
    parser.add_argument('--neg_factor', type=float, default=None, help='Negative factor for dominant event calculation')
    parser.add_argument('--pos_factor', type=float, default=None, help='Positive factor for dominant event calculation')
    parser.add_argument('--exp_factor', type=float, default=20, help='Initial exponential factor for cumsum calculation')

    args = parser.parse_args()

    main(args.light_spd_file, args.frame_statistics_spd_file, args.output_file, args.neg_factor, args.pos_factor, args.exp_factor)
