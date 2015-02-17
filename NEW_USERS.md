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
13. A screen will come up asking you to "Try Ubuntu" or "Install Ubuntu". Select "Install Ubuntu". Check the "Download updates while installing" box. Select "Replace... with Ubuntu".  

If you're using a Mac, you may have some success with some of the things you'll be required to install. If you're using Windows, you're SOL; [PETSc itself tells you not to use Windows for its software](http://www.mcs.anl.gov/petsc/documentation/installation.html#windows). 

If you're using OS X or Windows, here are your options:
- if you have a Mac mini in the office you're in, install Ubuntu 14.04 on that (best choice)
- install Ubuntu on a virtual machine (second best choice)
- install Ubuntu

If you're using the Mac mini in one of the offices, I would suggest installing Ubuntu 14.04 on it. If you have never installed formatted a drive and installed a new operating system (OS) on it, it's time to learn. 