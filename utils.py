
#create cosine simalarity function/part
def cosine_similarity():
        let magnitude = vDSP.sumOfSquares(A.scalars).squareRoot() * vDSP.sumOfSquares(B.scalars).squareRoot()
        let dotarray = vDSP.dot(A.scalars, B.scalars)
        return  dotarray / magnitude
    

