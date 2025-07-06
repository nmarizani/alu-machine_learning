-- Drop the function if it already exists to avoid conflicts
DROP FUNCTION IF EXISTS SafeDiv;

DELIMITER $$

-- Create a new SQL function called SafeDiv
CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS FLOAT
DETERMINISTIC
BEGIN
    -- If b is not zero, return a / b
    -- Else return 0 to avoid division by zero
    RETURN IF(b != 0, a / b, 0);
END$$

DELIMITER ;
