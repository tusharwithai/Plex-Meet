require('dotenv').config();
const { Client } = require('pg');
const fs = require('fs');
const path = require('path');

async function main() {
    const client = new Client({
        connectionString: process.env.DATABASE_URL,
        ssl: { rejectUnauthorized: false }
    });

    try {
        console.log("Connecting to Supabase Database (Node pg driver)...");
        await client.connect();

        console.log("Reading Prisma Schema SQL definitions...");
        const sql = fs.readFileSync(path.join(__dirname, 'prisma', 'init.sql'), 'utf8');

        console.log("Executing NextAuth & Room Schema mapping...");
        await client.query(sql);
        console.log("Schema injected successfully.");

        console.log("Applying Row Level Security to 'Room' table...");
        await client.query(`ALTER TABLE "Room" ENABLE ROW LEVEL SECURITY;`);
        
        console.log("Creating strict user-host insert policy for 'Room'...");
        // A simple RLS policy that ensures only the actual host can insert or manipulate their own room object.
        await client.query(`
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_policies WHERE tablename = 'Room' AND policyname = 'Users can only modify their own generated rooms'
                ) THEN
                    CREATE POLICY "Users can only modify their own generated rooms"
                    ON "Room"
                    FOR ALL
                    USING ("hostId" = current_setting('request.jwt.claims', true)::json->>'sub');
                END IF;
            END $$;
        `);
        console.log("RLS Policies enforced smoothly. ✅ Task 1 Pipeline Complete ✅");

    } catch (e) {
        console.error("Critical Execution Error: ", e);
    } finally {
        await client.end();
    }
}
main();
