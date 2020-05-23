import sys

newlines = []
added = {}
with open(sys.argv[1],"r") as f:
    for line in f:
        newlines.append(line)
        tokens=line.split()
        if(len(tokens)>=4):
            if(tokens[3] != "0"):
                if(added.get(tokens[0], None) == None):
                    added[tokens[0]] = True
                    newlines.append(f"{tokens[0]} {tokens[0]} 0 200004\n")
    print("".join(newlines))
            
