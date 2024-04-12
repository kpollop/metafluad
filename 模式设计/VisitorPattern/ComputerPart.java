package org.VisitorPattern;

public interface ComputerPart {
    public void accept(ComputerPartVisitor computerPartVisitor);
 }