import os
import pandas as pd
import click
import os
import re
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
from matplotlib.ticker import AutoMinorLocator

@click.group()
def main():
    """
    Main function to run the script
    """
    pass

def dataframe_to_dict(results, elements):
    """
    Convert DataFrame to dictionary with Element as keys and # Element as values,
    including elements with counts of 0.

    Args:
    results (DataFrame): DataFrame with 'Element' and '# Element' columns.
    elements (list): List of all elements to include in the dictionary.

    Returns:
    dict: Dictionary with Element as keys and # Element as values.
    """
    # Initialize an empty dictionary with all elements and counts set to 0
    d = {element: 0 for element in elements}

    # Iterate through DataFrame rows and update the counts
    for index, row in results.iterrows():
        d[row['Element']] = row['# Element']

    return d

def process_excel(file_path):
    """
    Process data from an Excel sheet
    """
    # Read the Excel file into a DataFrame
    data = pd.read_excel(file_path)
    return data

def process_cif_folder(folder_path):
    """
    Process data from a folder containing CIF files
    """
    entries = []
    formulas = []

    # Loop through the directory and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".cif"):
                # Extract the filename without extension
                entry = os.path.splitext(filename)[0]
                file_path = os.path.join(root, filename)

                # Read the file and extract the desired information
                with open(file_path, "r") as file:
                    # Keep track of which line we're processing
                    line_count = 0
                    for line in file:
                        line_count += 1
                        if line_count == 3:  # Check if it's the third line
                            # Split the line by '#'
                            parts = line.split("#")
                            if len(parts) > 2:
                                # Extract the second part, and remove leading/trailing whitespace
                                formula = parts[2].strip()
                                # Break the formula at the first space
                                formula = formula.split(" ")[0]
                                # Append the extracted data to the lists
                                entries.append(entry)
                                formulas.append(formula)
                            else:
                                click.secho(f"Warning: Line '{line}' in file '{filename}' does not contain enough '#' characters.", fg='yellow')
                            break  # Break the loop after finding the formula

    # Create a DataFrame from the lists
    data = pd.DataFrame({"Entry": entries, "Formula": formulas})
    return data

@main.command()
def run():
    """
    Run the main script
    """
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    available_files = os.listdir(script_dir)

    # List of elements
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
                'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


    # Filter out the "output" folder from the list of available directories
    available_folders = [folder for folder in available_files if os.path.isdir(os.path.join(script_dir, folder)) and folder not in ["output", "CIF_to_Excel"]]

    excel_sheets = [f for f in available_files if f.endswith('.xlsx')]
    cif_folders = [f for f in available_folders if os.path.isdir(os.path.join(script_dir, f))]

    if not excel_sheets and not cif_folders:
        click.secho("No Excel sheets or CIF folders available in the script's directory.", fg='cyan')
        return

    click.secho("Do you want to filter an Excel sheet/CIFs [1] or do you have them ready [2] (_Sorted.xlsx signifies they're ready)?", fg='cyan')
    choice = click.prompt("Enter the number corresponding to your choice", type=int)

    if choice == 1:
        click.secho("Available Excel sheets:", fg='cyan')
        for idx, sheet in enumerate(excel_sheets, start=1):
            click.echo(f"[{idx}] {sheet}")

        click.secho("Available CIF folders:", fg='cyan')
        for idx, folder in enumerate(cif_folders, start=len(excel_sheets)+1):
            click.echo(f"[{idx}] {folder}")

        choice = click.prompt("Enter the number corresponding to your choice", type=int)

        if 1 <= choice <= len(excel_sheets):
            file_path = os.path.join(script_dir, excel_sheets[choice - 1])
            data = process_excel(file_path)
            click.secho("Data processed from Excel sheet:", fg='cyan')
            click.echo(data)

            # Parse formulas and append elements and counts to DataFrame
            click.secho("Currently processing elements of your sheet", fg='cyan')

            # List of elements
            elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                        'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                        'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                        'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                        'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                        'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
                        'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

            # Function to parse the formula and extract elements
            def parse_formula(formula):
                elements_list = []
                counts_list = []
                current_element = ''
                current_count = ''
                for char in formula:
                    if char.isdigit() or char == '.':  # if character is a digit or a period
                        current_count += char
                    elif char.isupper():  # if character is uppercase letter
                        if current_element:
                            elements_list.append(current_element)
                            counts_list.append(float(current_count) if current_count and current_count != '.' else 1)
                            current_count = ''
                        current_element = char
                    elif char.islower():  # if character is lowercase letter
                        current_element += char
                # Add the last element and count
                if current_element:  
                    elements_list.append(current_element)
                    counts_list.append(float(current_count) if current_count and current_count != '.' else 1)
                return elements_list, counts_list

            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            data_copy = data.copy()

            # Apply the function to each row in the DataFrame
            data_copy[['Elements', 'Counts']] = data_copy['Formula'].apply(parse_formula).apply(pd.Series)

            # Split the lists into separate columns
            for i in range(max(map(len, data_copy['Elements']))):
                data_copy[f'Element {i+1}'] = data_copy['Elements'].str[i]
                data_copy[f'# Element {i+1}'] = data_copy['Counts'].apply(lambda x: x[i] if len(x) > i else None)

            # Drop temporary columns
            data_copy.drop(['Elements', 'Counts'], axis=1, inplace=True)

            click.secho("Elements and counts appended to DataFrame:", fg='cyan')
            click.echo(data_copy)

            # Save DataFrame to output folder
            output_folder = os.path.join(script_dir, "output")
            os.makedirs(output_folder, exist_ok=True)

            output_file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_Elements_Sorted.xlsx"
            output_file_path = os.path.join(output_folder, output_file_name)
            data_copy.to_excel(output_file_path, index=False)
            click.secho(f"Appended DataFrame saved to: {output_file_path}", fg='cyan')
        
        elif len(excel_sheets) < choice <= len(excel_sheets) + len(cif_folders):
            folder_path = os.path.join(script_dir, cif_folders[choice - len(excel_sheets) - 1])
            data = process_cif_folder(folder_path)
            click.secho("Data processed from CIF folder:", fg='cyan')
            click.echo(data)

            # Save raw data to Excel sheet if it is a CIF folder
            CIF_to_Excel_folder = os.path.join(script_dir, "CIF_to_Excel")
            os.makedirs(CIF_to_Excel_folder, exist_ok=True)

            raw_output_file_name = f"{os.path.basename(folder_path)}_Raw_Data.xlsx"
            raw_output_file_path = os.path.join(CIF_to_Excel_folder, raw_output_file_name)
            data.to_excel(raw_output_file_path, index=False)
            click.secho(f"Raw data saved to: {raw_output_file_path}", fg='cyan')

            # Parse formulas and append elements and counts to DataFrame
            click.secho("Currently processing elements of your sheet", fg='cyan')

            # List of elements
            elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                        'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                        'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                        'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                        'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                        'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
                        'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

            # Function to parse the formula and extract elements
            def parse_formula(formula):
                elements_list = []
                counts_list = []
                current_element = ''
                current_count = ''
                for char in formula:
                    if char.isdigit() or char == '.':  # if character is a digit or a period
                        current_count += char
                    elif char.isupper():  # if character is uppercase letter
                        if current_element:
                            elements_list.append(current_element)
                            counts_list.append(float(current_count) if current_count and current_count != '.' else 1)
                            current_count = ''
                        current_element = char
                    elif char.islower():  # if character is lowercase letter
                        current_element += char
                # Add the last element and count
                if current_element:  
                    elements_list.append(current_element)
                    counts_list.append(float(current_count) if current_count and current_count != '.' else 1)
                return elements_list, counts_list

            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            data_copy = data.copy()

            # Apply the function to each row in the DataFrame
            data_copy[['Elements', 'Counts']] = data_copy['Formula'].apply(parse_formula).apply(pd.Series)

            # Split the lists into separate columns
            for i in range(max(map(len, data_copy['Elements']))):
                data_copy[f'Element {i+1}'] = data_copy['Elements'].str[i]
                data_copy[f'# Element {i+1}'] = data_copy['Counts'].apply(lambda x: x[i] if len(x) > i else None)

            # Drop temporary columns
            data_copy.drop(['Elements', 'Counts'], axis=1, inplace=True)

            click.secho("Elements and counts appended to DataFrame:", fg='cyan')
            click.echo(data_copy)

            # Save DataFrame to output folder
            output_folder = os.path.join(script_dir, "output")
            os.makedirs(output_folder, exist_ok=True)

            output_file_name = f"{os.path.basename(folder_path)}_Elements_Sorted.xlsx"
            output_file_path = os.path.join(output_folder, output_file_name)
            data_copy.to_excel(output_file_path, index=False)
            click.secho(f"Appended DataFrame saved to: {output_file_path}", fg='cyan')

        else:
            click.secho("Invalid choice.", fg='red')



    elif choice == 2:
         output_folder = os.path.join(script_dir, "output")
    available_files = [file for file in os.listdir('output') if file.endswith('.xlsx') and not file.endswith("_errors.xlsx")]
    if not available_files:
        click.secho("No files found in the output folder.", fg='yellow')
    else:
        click.secho("Which file would you like to summarize:", fg='cyan')
        for idx, file_name in enumerate(available_files, start=1):
            click.echo(f"[{idx}] {file_name}")
        file_choice = click.prompt("Enter the number corresponding to your choice", type=int)
        if 1 <= file_choice <= len(available_files):
            chosen_file = os.path.join(output_folder, available_files[file_choice - 1])
            click.secho(f"Summarizing file: {chosen_file}", fg='cyan')

            # Find errors and view them within the screen:
            # Define a list of symbols that are not elements
            invalid_symbols = [char for char in set(''.join(elements)) if char.isalpha() and char.upper() not in elements and char != '.']

            # Define a DataFrame with invalid formulas
            invalid_formulas = pd.read_excel(chosen_file)

            # Function to parse the formula and extract elements
            def parse_formula(formula):
                elements_list = []
                counts_list = []
                error = None  # Initialize variable to store the error message
                current_element = ''
                current_count = ''
                for i, char in enumerate(formula, start=1):
                    if char.isdigit():  # if character is a number
                        current_count += char
                    elif char.isupper():  # if character is uppercase letter
                        if current_element:
                            if current_element.capitalize() in elements:  # Check both capital and lowercase versions
                                elements_list.append(current_element.capitalize())
                                counts_list.append(int(current_count) if current_count else 1)
                                current_count = ''
                                current_element = char
                            else:
                                error = f"'{current_element}' is not a valid element"
                                break
                        else:
                            current_element = char
                    elif char.islower():  # if character is lowercase letter
                        current_element += char
                    elif char in invalid_symbols:  # if character is an invalid symbol
                        error = f"'{char}' is not a valid symbol"
                        break
                    elif char == '.':  # Skip the '.' character
                        continue
                    else:  # if character is not recognized
                        error = f"'{char}' is not recognized"
                        break
                if current_element.capitalize() in elements:  # Check both capital and lowercase versions
                    elements_list.append(current_element.capitalize())
                    counts_list.append(int(current_count) if current_count else 1)
                else:
                    error = f"'{current_element}' is not a valid element"
                return elements_list, counts_list, error

            # Apply the function to each row in the DataFrame
            parsed_data = invalid_formulas['Formula'].apply(parse_formula).apply(pd.Series)
            invalid_formulas[['Elements', 'Counts', 'Error']] = parsed_data.iloc[:, :3]
        
            view_errors = 'y'  # Default to 'yes' without prompting
            # Would you like to view the errors?

            # Removed the prompt for viewing errors and directly set it to 'y'

            # Would you like to view the errors?
            # view_errors = click.prompt("Would you like to view the errors? [Y/n]", default='Y').strip().lower()

            if view_errors == 'y':
                # Filter the DataFrame for rows where the Error column is not None
                errors_df = invalid_formulas[invalid_formulas['Error'].notna()]
                click.secho("Errors found:", fg='red')
                if not errors_df.empty:
                    click.echo(errors_df)
                    # Save the errors to a file
                    error_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(chosen_file))[0]}_errors.xlsx")
                    errors_df.to_excel(error_file_path, index=False)
                    click.secho(f"Errors saved to: {error_file_path}", fg='cyan')
                else:
                    click.secho("No errors found in the DataFrame.", fg='green')
            else:
                click.secho("No errors found in the DataFrame.", fg='green')

            # Classification of formulas
            click.secho("Classifying your dataframe", fg='cyan')
            invalid_formulas_copy = invalid_formulas.copy()
            invalid_formulas_copy['System'] = None
            for index, row in invalid_formulas_copy.iterrows():
                num_elements = len(row['Elements'])
                if num_elements == 1:
                    invalid_formulas_copy.loc[index, 'System'] = 'Unary'
                elif num_elements == 2:
                    invalid_formulas_copy.loc[index, 'System'] = 'Binary'
                elif num_elements == 3:
                    invalid_formulas_copy.loc[index, 'System'] = 'Ternary'
                elif num_elements == 4:
                    invalid_formulas_copy.loc[index, 'System'] = 'Quaternary'

            # Save the summary to a new file
            summary_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(chosen_file))[0]}_Summary.xlsx")
            invalid_formulas_copy.to_excel(summary_file_path, index=False)
            click.secho(f"Summary saved to: {summary_file_path}", fg='cyan')
            
            # Filtering errors out of the dataframe
            click.secho("Filtering errors out of your dataframe", fg='cyan')
            filtered = invalid_formulas_copy[invalid_formulas_copy['Error'].isnull()]

            # Compiling the total number of elements in the dataframe
            click.secho("Compiling the total number of elements in your dataframe", fg='cyan')
            element_counts = {}
            for i in range(1, (len(filtered.columns) // 2) + 1):
                element_col = f'Element {i}'
                count_col = f'# Element {i}'
                if element_col not in filtered.columns or count_col not in filtered.columns:
                    continue
                for index, row in filtered.iterrows():
                    element = row[element_col]
                    count = row[count_col]
                    if pd.notnull(element) and pd.notnull(count):
                        if element in element_counts:
                            element_counts[element] += count
                        else:
                            element_counts[element] = count

            # Convert the dictionary to a DataFrame
            results = pd.DataFrame(list(element_counts.items()), columns=['Element', '# Element'])

            # Define the file path for saving the Excel file
            file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(chosen_file))[0]}_Element_Count.xlsx")

            # Save the DataFrame to an Excel file
            results.to_excel(file_path, index=False)

            # Print a message to confirm that the file has been saved
            click.secho(f"Element counts saved to: {file_path}", fg='cyan')

            # Print completion message
            click.secho("Element counting is completed", fg='cyan')

            # Print defining message
            click.secho("Converting dataframe to dictionary...", fg='cyan')

            # Convert DataFrame 'results' to dictionary using the function
            d = dataframe_to_dict(results, elements)

            formula = list(d.keys())


        # Creating your periodic table now
        click.secho("Creating your periodic table now", fg='cyan')

        # Define the element_prevalence function
        def element_prevalence(formulae,
                            name='ptable',
                            save_dir='output',
                            log_scale=False,
                            ptable_fig=True):
            
            desktop_path = os.path.expanduser("~/Desktop")  # Get the path to the desktop
            ptable_path = os.path.join(desktop_path, 'ptable.csv')  # Create the full path to ptable.csv
            
            ptable = pd.read_csv(ptable_path)

            ptable.index = ptable['symbol'].values
            elem_tracker = ptable['count']
            n_row = ptable['row'].max()
            n_column = ptable['column'].max()
            
            elem_tracker = pd.Series(data=d, index=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                        'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                        'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                        'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                        'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                        'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                        'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                        'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                        'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                        'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                        'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                        'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']) 

            
            
            if ptable_fig:
                fig, ax = plt.subplots(figsize=(n_column, n_row))
                rows = ptable['row']
                columns = ptable['column']
                symbols = ptable['symbol']
        #         print(symbols)
                rw = 1.0 # rectangle width (rw)
                rh = 1.0  # rectangle height (rh)
                for row, column, symbol in zip(rows, columns, symbols):
                    row = ptable['row'].max() - row
                    cmap = cm.GnBu                                           #Color
                    count_min = elem_tracker.min()
                    count_max = elem_tracker.max()
                    count_max = count_max + 24
        
                    norm = Normalize(vmin=count_min, vmax=count_max)
                    count = elem_tracker[symbol]
        
                    if log_scale:
                        norm = Normalize(vmin=np.log(1), vmax=np.log(count_max))
                        if count != 0:
                            count = np.log(count)
                    color = cmap(norm(count))
                    if count == 0:
                        color =   'white'
                    if count > 0 and count <= 10 :
                        color = 'lightyellow'    
                    rect = patches.Rectangle((column, row), rw, rh,
                                            linewidth=2,
                                            edgecolor='black',
                                            facecolor=color,
                                            alpha=1)

                    
                    plt.text(column+rw/2, row+rw/2, symbol,
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=20,
                            fontweight='semibold', color='black') 
                    ax.add_patch(rect)

                granularity = 20
                for i in range(granularity):
                    value = int(round((i) * count_max/(granularity-1)))
                    if log_scale:
                        if value != 0:
                            value = np.log(value)
                    color = cmap(norm(value))
                    if value == 0:
                        color = 'white' #white
                    if value > 0 and value <= 40 :
                        color = 'lightyellow'
                    length = 9
                    x_offset = 3.5
                    y_offset = 9.5 #7.8
                    x_loc = i/(granularity) * length + x_offset
                    width = length / granularity
                    height = 0.35
                    rect = patches.Rectangle((x_loc, y_offset), width, height,
                                            linewidth=1.5,
                                            edgecolor='gray',
                                            facecolor=color,
                                            alpha=1)

                    if i in [0, 4, 9, 14, 19]:
                        text = f'{value:0.0f}'
        #                 if i == 0:
        #                     text = 0
        #                 if i == 4:   #print(text)
        #                     text = nearest_100(i)
        #                 if i == 9:   #print(text)
        #                     text = 400
        #                 if i == 14:   #print(text)
        #                     text = 600
                        if log_scale:
                            text = f'{np.exp(value):0.1e}'.replace('+', '')
                        plt.text(x_loc+width/2, y_offset-0.4, text,
                                horizontalalignment='center',
                                verticalalignment='center',
                                fontweight='semibold',
                                fontsize=20, color='k')

                    ax.add_patch(rect)

                plt.text(x_offset+length/2, y_offset+0.7,
                        'log(Element Count)' if log_scale else 'Element Count',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontweight='semibold',
                        fontsize=20, color='k')

                ax.set_ylim(-0.15, n_row+.1)
                ax.set_xlim(0.15, n_column+4.5)

                # fig.patch.set_visible(False)
                ax.axis('off')

                
                if save_dir is not None:
                    base_name = os.path.basename(os.path.normpath(name))
                    file_name = f'{base_name}_ptable.png' if not base_name.endswith('_ptable') else f'{base_name}.png'
                    fig_name = os.path.join(save_dir, file_name)
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(fig_name, format='png', bbox_inches='tight', dpi=600)
                    
                plt.draw()
                plt.pause(0.001)
                plt.close()
                

        # Call the function with the list of elements and the relative path to the output directory
        element_prevalence(formula,
                        name=os.path.join('output'),
                        log_scale=False)
    
        click.secho("Periodic table created successfully", fg='cyan')

                # Prompt for filtering choice
        filter_choice = click.prompt("Would you like to filter based on either numerical or elemental composition? [Y/n]", type=str)

        if filter_choice.lower() == 'y':
            # Prompt for filtering type
            filtering_type = click.prompt("Numerical filtering [1] will separate your sorted dataframe based on unary/binary/ternary/quaternary entries, while elemental filtering [2] will remove entries with elements you don't want. Enter the corresponding number:", type=int)
            
            if filtering_type == 1:
                # Numerical filtering
                numerical_df = filtered.copy()

                # Group the DataFrame by the 'System' column and iterate over the groups
                for system, group in numerical_df.groupby('System'):
                    # Define the file name based on the system
                    file_name = os.path.join('output', f'numerical_{system.lower()}.xlsx')

                    # Save the grouped DataFrame to an Excel file
                    group.to_excel(file_name, index=False)

                    # Print a message to confirm the file has been saved
                    click.echo(f"Entries for {system} saved to: {file_name}")
            elif filtering_type == 2:
                # Elemental filtering
                # Prompt for elements to exclude
                elements_to_exclude = click.prompt("Enter elements to exclude (separated by commas), e.g., 'Rh, La, etc.':", type=str)
                elements_to_exclude = [elem.strip() for elem in elements_to_exclude.split(',')]

                # Elemental filtering logic
                elemental_df = invalid_formulas_copy.copy()

                # Initialize lists to store filtered and removed entries
                filtered_entries = []
                removed_entries = []

                # Loop through the DataFrame and filter entries
                for index, row in elemental_df.iterrows():
                    # Check if any of the specified elements are present in the row
                    if any(element in elements_to_exclude for element in row.values):
                        removed_entries.append(row)
                    else:
                        filtered_entries.append(row)

                # Create DataFrames from the filtered and removed entries
                elemental_filtered = pd.DataFrame(filtered_entries)
                elemental_removed = pd.DataFrame(removed_entries)

                # Optionally, you can reset the index of the DataFrames
                elemental_filtered.reset_index(drop=True, inplace=True)
                elemental_removed.reset_index(drop=True, inplace=True)

                # Define the file names for saving
                filtered_file = os.path.join('output', 'elemental_filtered.xlsx')
                removed_file = os.path.join('output', 'elemental_removed.xlsx')

                # Save the filtered DataFrame to an Excel file
                elemental_filtered.to_excel(filtered_file, index=False)

                # Save the removed DataFrame to an Excel file
                elemental_removed.to_excel(removed_file, index=False)

                # Print a message to confirm the files have been saved
                click.echo(f"Filtered entries saved to: {filtered_file}")
                click.echo(f"Removed entries saved to: {removed_file}")




@main.command()
@click.pass_context
def default(ctx):
    """
    Default command to run when no subcommand is provided
    """
    ctx.invoke(run)

if __name__ == "__main__":
    main()
