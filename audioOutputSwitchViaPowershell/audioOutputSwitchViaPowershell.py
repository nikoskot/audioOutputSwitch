import subprocess

audioOutputIndexes = [1, 2]


def run(cmd, text):
    completed = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=text)
    return completed


def getDefaultPlaybackDeviceIndex():
    # run Get-AudioDevice -Playback command to get default Playback device index
    output = run(cmd="Get-AudioDevice -Playback", text=True)

    # get index of default playback device
    index = [int(x[-1]) for x in output.stdout.split('\n') if x.startswith('Index')][0]

    return index


def setDefaultPlaybackDevice(index):
    # run Set-AudioDevice -Index command to set default Playback device based on the given index
    output = run(cmd="Set-AudioDevice -Index " + str(index), text=True)
    return output


def switchDefaultPlaybackDevice():
    # get current default playback device index
    currentIndex = getDefaultPlaybackDeviceIndex()

    # copy device indexes list
    audioOutputIndexesCopy = audioOutputIndexes.copy()

    # get other index
    audioOutputIndexesCopy.remove(currentIndex)

    otherIndex = audioOutputIndexesCopy[0]

    output = setDefaultPlaybackDevice(otherIndex)


if __name__== '__main__':
    switchDefaultPlaybackDevice()