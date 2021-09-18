import pixy

def gradient(colour_a, colour_b):
    
    output = []

    for i in range(0, 100, 2): 

        p = i / 100 

        output.append((
            int(colour_a[0] + p * (colour_b[0] - colour_a[0])),
            int(colour_a[1] + p * (colour_b[1] - colour_a[1])),
            int(colour_a[2] + p * (colour_b[2] - colour_a[2]))
        ))

    return output

red_to_green = gradient((255, 0, 0), (0, 255, 0)) 
green_to_blue = gradient((0, 255, 0), (0, 0, 255))
blue_to_red = gradient((0, 0, 255), (255, 0, 0)) 

for colour in red_to_green:
    print(pixy.pring(" ", pixy.TrueColour(*colour, background=True)), end="")

print()

for colour in green_to_blue:
    print(pixy.pring(" ", pixy.TrueColour(*colour, background=True)), end="")

print()

for colour in blue_to_red:
    print(pixy.pring(" ", pixy.TrueColour(*colour, background=True)), end="")

print()