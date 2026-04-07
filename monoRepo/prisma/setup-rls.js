const { PrismaClient } = require('@prisma/client')
const prisma = new PrismaClient()

async function main() {
  console.log("Enabling Row Level Security on Room table...");
  try {
    await prisma.$executeRawUnsafe(`ALTER TABLE "Room" ENABLE ROW LEVEL SECURITY;`);
    console.log("Successfully enabled RLS!");
  } catch(e) {
    console.error("Error setting RLS:", e);
  } finally {
    await prisma.$disconnect()
  }
}

main()
