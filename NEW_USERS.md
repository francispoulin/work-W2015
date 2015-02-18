# Getting Started
This is a guide aimed at new students of Professor Francis Poulin working on the same project I did. I'm creating this guide because it took me a bit of time to get started because the resources out there are either not that great, or hard to find (or both!).


## Operating System installation
This guide assumes you'll be using a recent version of Ubuntu (i.e. 12.04, 14.04). You might be asking, "what the hell is Ubuntu?" Ubuntu is a Linux-based operating system (OS). It's like Windows XP/7/8 or Mac OS X in the sense that you can run computer programs on it. As to what "Linux" is, that's a bit more complicated and not something I'll explain. If you have a Mac mini in your office, read the next section. If you are planning on using your own laptop and do not already have Ubuntu on it, then skip to [Alternative Options](#AlternativeOptions)

### Installing Ubuntu 14.04 on a Mac mini
Ideally, that Mac mini was the one I used and already has Ubuntu installed on it. If it doesn't, then I would really recommend installing Ubuntu on it. This requires formatting the hard drive (i.e. deleting everything on the computer), so if you think this is going to be a problem for the last person that used the computer, you should contact them.

The Mac mini is most likely a late 2012 model, so I would suggest [following this guide to install Ubuntu](https://theredblacktree.wordpress.com/2014/07/29/installation-guide-for-linux-mint-17-ubuntu-14-04-on-apple-mac-mini-late-2012/). If you don't understand what's going on in this guide, I'll explain a bit:

0.	Open up the Terminal application by going to Finder > Applications > Utilities > Terminal
1.	Download Ubuntu 14.04 from [the official Ubuntu website](http://releases.ubuntu.com/14.04/). Choose Desktop Image > 64-bit PC (AMD64) desktop image.
2.	Move the file you just downloaded into the home directory and rename it `linux.iso` . Your home directory can be accessed by opening up the Finder and pressing `Cmd+Shift+H`.
3.	In the Terminal, type `hdiutil convert -format UDRW -o ~/linux.img ~/linux.iso` and press `Enter`
4.	There should be a new file in your directory. If it is called `linux.img.dmg`, rename it to be `linux.img`.
5.	Plug in a USB stick that is at least 2 GB in size.
6.	In the Terminal, type `diskutil list` and press `Enter`. Look for where it says `usb_disk`. Almost immediately above that, it should say something like `/dev/diskN`, where N is a number. Make note of this. (Check the picture in the guide for clarification).
7.	Type `diskutil unmountDisk /dev/diskN` where N is the number of the USB stick you wrote down and press `Enter`.
8.	Type `sudo dd if=~/linux.img of=/dev/rdiskN bs=1m` where N is the number of the USB stick and press `Enter`. This step may take a *while*, so feel free to screw around for a bit.
9.	After the previous step is done its thing, type `diskutil eject /dev/diskN` and press `Enter`. Wait 5 seconds and then pull the USB stick out.
10.	Reboot the computer. Keep pressing the `Option (Alt)` key while it is booting.
11. Enter the administration password, if a screen comes up that requires it.
12. You will end up in a bootloader screen for the Mac. Plug in your USB and press `Enter`.
13. A screen will come up asking you to "Try Ubuntu" or "Install Ubuntu". Select "Install Ubuntu". Check the "Download updates while installing" box. Select "Replace... with Ubuntu". The rest of the steps should be simple enough.
14. TODOOOOOOOO

### Installing Ubuntu on your own computer
If you're using a Mac, you may have some success with some of the things you'll be required to install. If you're using Windows, you're SOL; [PETSc itself tells you not to use Windows for its software](http://www.mcs.anl.gov/petsc/documentation/installation.html#windows). Either way, I'm going to give you your options on installing Ubuntu:

- install Ubuntu on a virtual machine (recommended)
- install Ubuntu alongside your current OS (not recommended unless you know what you're doing)

A virtual machine (VM) is an emulation of an OS - it allows you to run an OS inside your current OS. **When trying to measure the performance of algorithms you are working with, do not run the tests inside your VM.** It will not give accurate results. The workaround to this is using the computers accessible to researchers. More on this later.

To install Ubuntu on a VM, you must first install a program that allows you to create a VM. VirtualBox is good for this. [Follow this guide to properly install Ubuntu on a VM using VirtualBox](http://www.wikihow.com/Install-Ubuntu-on-VirtualBox). It's pretty straightforward. (The guide is for Windows, but I'm sure it's very similar for OS X).

## Installing the iPython Notebook
"The IPython Notebook is a web-based interactive computational environment where you can combine code execution, text, mathematics, plots and rich media into a single document." It's a very neat tool does exactly that. I'll be using it for those reasons exactly - there are many equations to write and putting them in here wouldn't look very nice. So, let's install it. Open up a terminal (`Ctrl+Alt+T`).

We need to install some libraries before iPython itself. Type in (or copy and paste) the following into the terminal:

```
sudo apt-get update
sudo apt-get install python-dev -y
sudo apt-get install python-pip
sudo pip install "ipython[notebook]"
```

This updates the libraries you already have, installs *python-dev* which is required for iPython's dependencies, installs *pip* which is a Python module that makes it easier to install other Python modules (how meta), and then uses pip to install iPython and the iPython notebook. To run the iPython notebook, type in:

```
ipython notebook
```

(Makes sense, eh?)
