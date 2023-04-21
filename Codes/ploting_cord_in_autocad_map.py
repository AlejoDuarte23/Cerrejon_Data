import os
import pandas as pd
import pyautocad
import math

def within_circle(x, y, circle_center, circle_radius):
    distance = math.sqrt((circle_center[0] - x) ** 2 + (circle_center[1] - y) ** 2)
    return distance <= circle_radius

def Loading_file(sheet_names,file_route_name):
    cols_of_interest = ['Timestamp', 'PositionX', 'PositionY']
    df = pd.concat(pd.read_excel(file_route_name,
                                 sheet_name=sheet_names, usecols=cols_of_interest
                                 , skiprows=[0]), ignore_index=True)
    # Convert PositionX and PositionY to integers (if needed)
    df['PositionX'] = df['PositionX'].astype(int)
    df['PositionY'] = df['PositionY'].astype(int)
    return df

# Tajos Circle centers 
tajo_tabaco_puente = [1166343.4113,1723880.6997]
tajo_ANNEX = [1157678.4946,1714565.5789]
tajo_patilla_1 = [1155328.7377,1720141.998]
tajo_patilla_2 = [1150385.1278,1715671.3838]

circle_centers_radii = [
    (tajo_tabaco_puente, 4430.6225, 5, 'tajo_tabaco_puente'),  # blue
    (tajo_ANNEX, 2627.4178, 6, 'tajo_ANNEX'),  # red
    (tajo_patilla_1, 3180.828, 1, 'tajo_padilla'),  # red
    (tajo_patilla_2, 7180.6501, 1, 'tajo_padilla')  # red
]

def plot_cords_autocad(df,PLT = False):
    df['location_id'] = ''
    if PLT == True:
        acad = pyautocad.Autocad()

    # Loop over the coordinates and add circles to the model space
    for i, row in df.iterrows():
        x, y = row['PositionX'], row['PositionY']
    
        # Default color and radius and id
        color = 2  # yellow
        radius = 50
        location_id = 'Roads'
        
        for circle_center, circle_radius, circle_color, circle_id in circle_centers_radii:
            if within_circle(x, y, circle_center, circle_radius):
                color = circle_color
                radius = radius
                location_id = circle_id
                break
    
        # Assign the location_id to the DataFrame
        df.at[i, 'location_id'] = location_id
        if PLT== True:
            center = pyautocad.APoint(x, y, 0)
            circle = acad.model.AddCircle(center, radius)
            circle.color = color
    
    # Save the changes
    acad.doc.Save()


#  inputs

file_name = 'Datos GPS equipo 022-429 - Mayo-22 hasta Abril-5 2023.xlsx'
file_route_name = os.path.join('..','Reference',file_name)

# file_route_name = 'Datos GPS equipo 022-429 - Mayo-22 hasta Abril-5 2023.xlsx'

sheet_names = ['20230322',
               '20230323',
               '20230324',
               '20230325',
               '20230326',
               '20230327',
               '20230328',
               '20230329',
               '20230330']

df = Loading_file(sheet_names,file_route_name)
plot_cords_autocad(df)


