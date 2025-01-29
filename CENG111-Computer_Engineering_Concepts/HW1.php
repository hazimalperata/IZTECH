<?php
$A = 0;
$B = 0;
$C = 0;
$D = 0;
$E = 0;
$F = 0;
$a = 0;	
$b = 0;
	?>
<html>
<head>
	<title>Definite Integral Calculation</title>
</head>
<body>
	<h1>Definite Integral Calculation</h1>
	<form method="POST">
<div>
f(x)=
<input name="A" value=<?=$A?> size = '2'>+
<input name="B" value=<?=$B?> size = '2'>x+
<input name="C" value=<?=$C?> size = '2'>x<sup>2</sup>+
<input name="D" value=<?=$D?> size = '2'>x<sup>3</sup>+
<input name="E" value=<?=$E?> size = '2'>exp(x)+
<input name="F" value=<?=$F?> size = '2'>sin(x)
</div>
<br>
<div>
a=<input name="a" value=<?=$a?> size = '2'>
b=<input name="b" value=<?=$b?> size = '2'>
</div>
<br>
<button type="submit">Submit</button>
</form>




</body>
</html>
<?php
if ($_POST) {
	$A = $_POST['A'];
	$B = $_POST['B'];
	$C = $_POST['C'];
	$D = $_POST['D'];
	$E = $_POST['E'];
	$F = $_POST['F'];
	$a = $_POST['a'];	
	$b = $_POST['b'];
	$firstPart = ($_POST['A'] * $_POST['b']) - ($_POST['A'] * $_POST['a']);
	$secondPart = (($_POST['B']/2) * ($_POST['b']**2)) - (($_POST['B']/2) * ($_POST['a']**2));
	$thirdPart = (($_POST['C']/3) * ($_POST['b']**3)) - (($_POST['C']/3) * ($_POST['a']**3));
	$fourthPart = (($_POST['D']/4) * ($_POST['b']**4)) - (($_POST['D']/4) * ($_POST['a']**4));
	$expPart = $_POST['E'] * (exp($_POST['b'])-exp($_POST['a']));
	$sinPart = -1 * ($_POST['F'] * (cos($_POST['b'])-cos($_POST['a'])));
echo "".$firstPart+$secondPart+$thirdPart+$fourthPart+$expPart+$sinPart ;}
?>