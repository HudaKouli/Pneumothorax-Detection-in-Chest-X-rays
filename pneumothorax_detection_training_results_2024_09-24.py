import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'
lables_properties = {'family':'Times New Roman', 'size': 12}
file_name = 'pneumothorax_detection_training_results_2024_09-24.xlsx'

# Function to plot the results
def plot_results(file_name):
    try:
        df = pd.read_excel(file_name)
    except FileNotFoundError:
        print(f"File {file_name} not found. Please check the file path and try again.")
        return

    # Check if required columns exist
    required_columns = ["Epoch", "Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy"]
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns in {file_name}. The file should contain: {required_columns}")
        return

    # Plotting
    epochs_range = range(1, len(df['Train Loss']) + 1)
    plt.figure(figsize= (14,7.5),dpi = 100)
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', marker='o', linestyle = '--', color=plt.cm.viridis(0.2),linewidth = 2)
    plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss', marker='o',linestyle = '--',color=plt.cm.viridis(0.5),linewidth = 2)
    plt.xlabel('Epoch', lables_properties )
    plt.ylabel('Loss',lables_properties)
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper left',bbox_to_anchor=(1, 1), 
        framealpha=0.5, 
        facecolor='white', 
        edgecolor='black',prop={'size': 12,'family':'Times New Roman' })
    plt.xlim([0, max(epochs_range)]) 
    plt.grid(True,which='both', color='lightgray', linestyle='--')

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy', marker='o',linestyle = '--',color=plt.cm.viridis(0.2),linewidth = 2)
    plt.plot(df['Epoch'], df['Val Accuracy'], label='Val Accuracy', marker='o',linestyle = '--',color=plt.cm.viridis(0.5),linewidth = 2)
    plt.xlabel('Epoch', lables_properties)
    plt.ylabel('Accuracy', lables_properties)
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='upper left',bbox_to_anchor=(1, 1), 
        framealpha=0.5, 
        facecolor='white', 
        edgecolor='black',prop={'size': 12,'family':'Times New Roman' })
    plt.xlim([0, max(epochs_range)]) 

    # Add grid lines with light color, thin lines, and dashed style
    plt.grid(True, which='both', color='lightgray', linestyle='--')

    # Show plots
    plt.tight_layout()
    # Save the plot as an SVG file
    plt.savefig("training_plot.svg", format="svg")
    plt.show()


#calling the plot function.
plot_results(file_name)