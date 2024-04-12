package org.VisitorPattern;

public class Monitor  implements ComputerPart {
 
    @Override
    public void accept(ComputerPartVisitor computerPartVisitor) {
       computerPartVisitor.visit(this);
    }
 }