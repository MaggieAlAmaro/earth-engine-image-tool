import os, sys, time


#WARNING, MUST include a target keyword
def processDirOrFile(function, **kwargs):
    target = kwargs.get('target')
    destination = kwargs.get('destination',None)

    if(destination == None and target != None):
        if(os.path.isdir(target)):
            for filename in os.listdir(target):
                f = os.path.join(target, filename)
                if os.path.isfile(f):
                    function(f, **kwargs)
        elif os.path.isfile(target):
            function(target, **kwargs)
        else:
            print("Target not found. Check spelling.")
    
    elif(os.path.isdir(target) and os.path.isdir(destination)):
        for target_file, destination_file in zip(os.listdir(target), os.listdir(destination)):
            target_f = os.path.join(target, target_file)
            destination_f = os.path.join(destination, destination_file)
            if os.path.isfile(target_f) and os.path.isfile(destination_f):
                function(target_f,destination_f, **kwargs)
            else:
                print("Target not found. Check spelling.")
    
    elif(os.path.isfile(target) and os.path.isfile(destination)):
        function(target,destination, **kwargs)
    else:
        print("Wrong usage.")


# Creates new filename from a given filename
# WARNING: suffix MUST hold file extention
def newFilename(oldname, prefix=None, suffix=".png", outdir='.'):
    f_type = os.path.splitext(oldname)[-1]
    newFileName = os.path.basename(oldname)
    noExtension = newFileName.rstrip(f_type)
    fn = noExtension
    if prefix != None:        
        fn = prefix + noExtension
    fn = fn + suffix
    fn = os.path.join(outdir,fn) 
    #fn = os.path.join(os.path.dirname(oldname),fn) 
    print("New filename: " + fn)
    return fn



def makeOutputFolder(folderFunctionName):
    return makeLogFolder('output',folderFunctionName)


def makeLogFolder(logFolderName, folderFunctionName):
    output = os.path.join(logFolderName, (folderFunctionName + "-" + time.strftime("%Y-%m-%d-%H-%M")))
    os.mkdir(output)
    return output
