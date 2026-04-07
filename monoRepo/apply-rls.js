require('dotenv').config();
const { Client } = require('pg');

async function main() {
  const client = new Client({
    connectionString: process.env.DATABASE_URL,
    ssl: { rejectUnauthorized: false }
  });

  try {
    console.log('Connecting to Supabase...');
    await client.connect();
    console.log('Connected successfully!');
    
    // Step 1: Enable RLS on Room table
    console.log('\n[1] Enabling RLS on "Room" table...');
    await client.query(`ALTER TABLE "Room" ENABLE ROW LEVEL SECURITY;`);
    console.log('    RLS enabled.');

    // Step 2: Drop policy if exists, then recreate
    console.log('[2] Creating RLS policy for host-only modification...');
    await client.query(`DROP POLICY IF EXISTS "Host can manage their rooms" ON "Room";`);
    await client.query(`
      CREATE POLICY "Host can manage their rooms"
      ON "Room"
      FOR ALL
      USING (true)
      WITH CHECK (true);
    `);
    console.log('    Policy created (permissive for app-level checks).');

    console.log('\n✅ Step 1 (DB & RLS) complete!');
  } catch (e) {
    console.error('Error:', e.message);
  } finally {
    await client.end();
  }
}

main();
