import os, sys

def processDirOrFile(function, **kwargs):
    target = kwargs.get('target')
    destination = kwargs.get('destination',None)
    out = kwargs.get('out', None)

    if(destination == None and target != None):
        if(os.path.isdir(target)):
            for filename in os.listdir(target):
                f = os.path.join(target, filename)
                if os.path.isfile(f):
                    function(f,out)
        elif os.path.isfile(target):
            function(target,out)
    
    elif(os.path.isdir(target) and os.path.isdir(destination)):
        for target_file, destination_file in zip(os.listdir(target), os.listdir(destination)):
            target_f = os.path.join(target, target_file)
            destination_f = os.path.join(destination, destination_file)
            if os.path.isfile(target_f) and os.path.isfile(destination_f):
                function(target_f,destination_f,out)
    
    elif(os.path.isfile(target) and os.path.isfile(destination)):
        function(target,destination,out)

