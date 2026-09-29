#Splitting the signal into 10ms long frames
def framing(y, sr, Tf):
    Nf = int(sr*Tf)
    frames = [ (y[i*Nf:(i+1)*Nf], sr) for i in range(len(y)//Nf) ]
    return frames