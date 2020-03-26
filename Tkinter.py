#Code for Capillary Action Tool made .exe file for Mechanical Students
import tkinter
import math
from tkinter import *

root = tkinter.Tk()
root.geometry("800x800")
root.title("Capillary Action")


frame1 = Frame(root)
frame1.grid(sticky=W)
L1 = Label(frame1, text="Surface Tension (N/m)")
L1.grid(row=0, column=1)
S1 = Scale(frame1, from_=0, to=0.1, resolution=0.0001, orient=HORIZONTAL, activebackground="orange")
S1.grid(row=0, column=2)
E1 = Entry(frame1, bd=5, highlightcolor="blue", width=10)
E1.grid(row=0, column=3)

frame2 = Frame(root)
frame2.grid(sticky=W)
L2 = Label(frame2, text="Contact Angle (°)")
L2.grid(row=1, column=1)
S2 = Scale(frame2, from_=10, to=90, resolution=0.1, orient=HORIZONTAL, activebackground="orange")
S2.grid(row=1, column=2)
E2 = Entry(frame2, bd=5, highlightcolor="blue", width=10)
E2.grid(row=1, column=3)

frame3 = Frame(root)
frame3.grid(sticky=W)
L3 = Label(frame3, text="Density")
L3.grid(row=2, column=1)
S3 = Scale(frame3, from_=500, to=2000, resolution=0.1, orient=HORIZONTAL, activebackground="orange")
S3.grid(row=2, column=2)
E3 = Entry(frame3, bd=5, highlightcolor="blue", width=10)
E3.grid(row=2, column=3)

frame4 = Frame(root)
frame4.grid(sticky=W)
L4 = Label(frame4, text="Capillary radius (mm)")
L4.grid(row=3, column=1)
S4 = Scale(frame4, from_=0.5, to=6.5, resolution=0.01, orient=HORIZONTAL, activebackground="orange")
S4.grid(row=3, column=2)
E4 = Entry(frame4, bd=5, highlightcolor="blue", width=10)
E4.grid(row=3, column=3, sticky=E)

frame5 = Frame(root)
frame5.grid(sticky=W)
label1 = Label(frame5, text="Height (mm)=")
label1.grid(row=4, column=1)
E5 = Entry(frame5, bd=5, highlightcolor="blue")
E5.grid(row=4, column=2)


def scale(event):
    S1.set(float(E1.get()))
    S2.set(float(E2.get()))
    S3.set(float(E3.get()))
    S4.set(float(E4.get()))
    combine_funcs(event)


def cal_height(event):
    E1.delete(0, END)
    E1.insert(0, S1.get())
    gamma = float(S1.get())
    angle = float(S2.get())
    theta = math.cos(math.radians(angle))
    rho = float(S3.get())/1000
    radius = float(S4.get())
    a = 2*gamma*theta
    b = rho*radius*9.81
    E5.delete(0, END)
    h = (a/b)*1000
    height = round(h, 2)
    E5.insert(0, height)
    E5.grid(row=4, column=2)


C = tkinter.Canvas(root, bg="white", height=465, width=400)


def diagram(event):
    E1.delete(0, END)
    E1.insert(0, S1.get())
    gamma = float(S1.get())
    E2.delete(0, END)
    E2.insert(0, S2.get())
    angle = float(S2.get())
    theta = math.cos(math.radians(angle))
    E3.delete(0, END)
    E3.insert(0, S3.get())
    rho = float(S3.get())/1000
    E4.delete(0, END)
    E4.insert(0, S4.get())
    rad = float(S4.get())
    a = 2 * gamma * theta
    b = rho * rad * 9.81
    h = float((a / b) * 1000)
    height = round(h, 2)
    C.delete("all")
    coord7 = 190.5 + rad * 10, 410, 190.5 + (rad * 10), 410 - height*5 + theta, 190, 410 - height*5 + (
            8 * theta), 189.5 - rad * 10, 410 - height*5 + theta, 189.5 - rad * 10, 410
    C.create_polygon(coord7, fill="blue")
    # water line left side
    coord6 = 189.5-rad*10, 410, 179.5-rad*10, 415, 20, 415, 10, 410, 10, 460, 390, 460, 390, 410, 380, 415, 200.5+rad*10, \
             415, 190.5+rad*10, 410
    C.create_polygon(coord6, fill="light blue")

    # vertical radius line
    rad = float(S4.get())
    coord4 = 189.5 - rad * 10, 5, 189.5 - rad * 10, 440
    line4 = C.create_line(coord4, fill="red")
    coord5 = 190.5 + rad * 10, 4, 190.5 + rad * 10, 440
    line5 = C.create_line(coord5, fill="red")
    coord1 = 10, 380, 10, 460
    line1 = C.create_line(coord1, fill="red")
    coord2 = 390, 380, 390, 460
    line2 = C.create_line(coord2, fill="red")
    coord3 = 10, 460, 390, 460
    line3 = C.create_line(coord3, fill="red")
    C.grid(row=6, column=0)


def combine_funcs(event):
    diagram(event)
    cal_height(event)


E1.bind('<Return>', scale)
E2.bind('<Return>', scale)
E3.bind('<Return>', scale)
E4.bind('<Return>', scale)

S1.bind('<ButtonRelease-1>', combine_funcs)
S2.bind('<ButtonRelease-1>', combine_funcs)
S3.bind('<ButtonRelease-1>', combine_funcs)
S4.bind('<ButtonRelease-1>', combine_funcs)

C1 = tkinter.Canvas(root, bg="white", height=655, width=400)
coordA = 10, 5, 10, 190, 390, 190, 390, 5, 10, 5
lineA = C1.create_line(coordA, fill="black")
coordB = 76, 5, 76, 190, 240, 190, 240, 5
lineB = C1.create_line(coordB, fill="black")
coordC = 10, 35, 390, 35
lineC = C1.create_line(coordC, fill="black")
# Fluids Name
fluids = C1.create_text(35, 20, text="Fluids")
Water = C1.create_text(35, 50, text="Water")
Alcohol = C1.create_text(38, 80, text="Alcohol")
Diesel = C1.create_text(35, 110, text="Diesel")
Milk = C1.create_text(35, 140, text="Milk")
Petrol = C1.create_text(35, 170, text="Petrol")
# Specific Gravity
fluids1 = C1.create_text(160, 20, text="Surface Tension (N/m)")
Water1 = C1.create_text(150, 50, text="71.99")
Alcohol1 = C1.create_text(150, 80, text="22.39")
Diesel1 = C1.create_text(150, 110, text="0.00003")
Milk1 = C1.create_text(150, 140, text="47.5")
Petrol1 = C1.create_text(150, 170, text="0.00002")
# Density
fluids3 = C1.create_text(320, 20, text="Density (kg/m^3)")
Water3 = C1.create_text(310, 50, text="997")
Alcohol3 = C1.create_text(310, 80, text="789")
Diesel3 = C1.create_text(310, 110, text="832")
Milk3 = C1.create_text(310, 140, text="1035")
Petrol3 = C1.create_text(310, 170, text="719.7")

# Theory
C1.create_text(210, 409, text="Capillary action is important for moving water \n (and all of the things that are "
                              "dissolved in it) around. \nIt is defined as the movement of water within the spaces \n of"
                              " aporous material due to the forces of adhesion,\n cohesion, and surface tension. \n  "
                              "Capillary action occurs because water is sticky, \n thanks to the forces of cohesion "
                              "(water molecules like \n to stay close together) and adhesion (water molecules are \n "
                              "attracted and stick to other substances). Adhesion of \n water to the walls of a vessel "
                              "will cause \n an upward force on the liquid at the edges and result \n in a meniscus "
                              "which turns upward. The surface tension acts \n to hold the surface intact. Capillary"
                              " action occurs when the \n adhesion to the walls is stronger than the cohesive forces"
                              " \nbetween the liquid molecules. The height to which capillary \naction will take "
                              "water in a uniform circular tube (picture to right) is \n limited by surface tension and"
                              "of course, gravity. \n   Liquids (such as water) that wet glass climb upward on the "
                              "\nsurfaces of their containers to form a concave meniscus. \n This occurs when  "
                              "adhesive solid-liquid intermolecular forces are \nstronger than liquid forces. Such "
                              "liquids will rise in a \n narrow capillary tube until a balance is established \n between "
                              "the effects of surface tension \n and gravity. "
                              "The capillary rise increases sharply as the tube is \n made narrower. For example, "
                              "water in a glass capillary of radius \n 0.1 mm will rise by about 140 mm . \n "
                              "The Formula used is : (2γ cosθ/ρgr) where \n γ=Surface tension \n ρ=Specific gravity "
                              "and r=Radius of Capillary tube ")
C1.place(x=400, y=0)

# Image
C2 = tkinter.Canvas(root, height=345, width=400)
photo = """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"""
photo = PhotoImage(data=photo)
image = C2.create_image(5, 8, image=photo, anchor=NW)
C2.place(x=400, y=630)

frame6 = Frame(root, width=400, height=100)
frame6.place(x=0, y=660)
label2 = Label(frame6, text="Capillary Action Tool", font="Times", fg="maroon")
label2.place(x=100, y=10)
root.mainloop()
