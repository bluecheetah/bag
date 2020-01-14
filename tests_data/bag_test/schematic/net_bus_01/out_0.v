


module nmos4_18(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module nmos4_svt(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module nmos4_lvt(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module nmos4_hvt(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module nmos4_standard(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module nmos4_fast(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module nmos4_low_power(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module pmos4_18(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module pmos4_svt(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module pmos4_lvt(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module pmos4_hvt(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module pmos4_standard(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module pmos4_fast(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule

module pmos4_low_power(
    inout B,
    inout D,
    inout G,
    inout S
);
endmodule


module pin_array_0_1(
    input  wire VDD,
    input  wire VSS,
    input  wire [3:0] vin,
    output wire vout
);

endmodule


module pin_array_0_2(
    input  wire VDD,
    input  wire VSS,
    input  wire vin,
    output wire vout
);

endmodule


module pin_array_0(
    input  wire VDD,
    input  wire VSS,
    input  wire vin,
    output wire [3:0] mid,
    output wire vout
);

pin_array_0_2 X0_3 (
    .VDD( VDD ),
    .VSS( VSS ),
    .vin( vin ),
    .vout( mid[3] )
);

pin_array_0_2 X0_2 (
    .VDD( VDD ),
    .VSS( VSS ),
    .vin( vin ),
    .vout( mid[2] )
);

pin_array_0_2 X0_1 (
    .VDD( VDD ),
    .VSS( VSS ),
    .vin( vin ),
    .vout( mid[1] )
);

pin_array_0_2 X0_0 (
    .VDD( VDD ),
    .VSS( VSS ),
    .vin( vin ),
    .vout( mid[0] )
);

pin_array_0_1 X1 (
    .VDD( VDD ),
    .VSS( VSS ),
    .vin( mid[3:0] ),
    .vout( vout )
);

endmodule
