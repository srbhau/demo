#Diffie Hellman Key Exchange Alogorithm
# Begin
sharedPrime = int(input("Enter shared Prime(n):"))
sharedBase = int(input("Enter shared Base(g):"))
aliceSecret = int(input("Enter Alice Secret Key(x):"))
bobSecret = int(input("Enter Bob Secret Key(y):"))
print( "Publicly Shared Variables:")
print( "    Publicly Shared Prime: " , sharedPrime )
print( "    Publicly Shared Base:  " , sharedBase )
 
# Alice Sends Bob A = g^x mod n
A = (sharedBase**aliceSecret) % sharedPrime
print( "\n  Alice Sends(A) Over Public Chanel: " , A )
 
# Bob Sends Alice B = g^y mod n
B = (sharedBase ** bobSecret) % sharedPrime
print("Bob Sends(B) Over Public Chanel: ", B )
 
print( "\n------------\n" )
print( "Privately Calculated Shared Secret:" )
# Alice Computes Shared Secret: K1 = B^x mod n
aliceSharedSecret = (B ** aliceSecret) % sharedPrime
print( "    Alice Shared Secret: ", aliceSharedSecret )
 
# Bob Computes Shared Secret: K2 = A^y mod n
bobSharedSecret = (A**bobSecret) % sharedPrime
print( "    Bob Shared Secret: ", bobSharedSecret )

